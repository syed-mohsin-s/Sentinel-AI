"""
Tier 2 Edge SLM Judge
Executes local SLMs (Qwen2.5 / Gemma 3 / Llama 3.2) via HuggingFace Transformers,
Ollama, or ONNX Runtime / QNN Execution Provider.
Generates structured JSON coercion reasoning.
"""

import json
import time
import urllib.request
from typing import Dict, Optional


class Tier2SLMJudge:
    """
    Tier 2 Edge SLM Reasoning Engine.
    Supports live HuggingFace Transformers pipeline, Ollama API, ONNX Runtime,
    and fast NPU heuristic fallback.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_npu: bool = False,
        backend: str = "auto"  # Options: "auto", "transformers", "ollama", "heuristic"
    ):
        self.model_path = model_path or "Qwen/Qwen2.5-0.5B-Instruct"
        self.use_npu = use_npu
        self.backend = backend
        self.pipeline = None
        self.is_loaded = False
        self.active_backend = "Heuristic (NPU Simulator)"

        self.system_prompt = (
            "You are Sentinel AI, an edge-native real-time fraud judge. "
            "Analyze the call transcript for social engineering, coercion, or authority impersonation. "
            "Output strict JSON only with keys: 'is_scam' (bool), 'confidence' (float 0-1), "
            "'threat_type' (str), 'intent_summary' (str), 'recommended_action' (str)."
        )

    def load_model(self):
        """Attempts to load a live local LLM backend if available."""
        if self.is_loaded:
            return

        # 1. Check Ollama API endpoint
        if self.backend in ["auto", "ollama"]:
            try:
                req = urllib.request.Request("http://localhost:11434/api/tags")
                with urllib.request.urlopen(req, timeout=0.5) as resp:
                    if resp.status == 200:
                        self.active_backend = "Ollama (Local LLM API)"
                        self.is_loaded = True
                        return
            except Exception:
                pass

        # 2. Check Transformers Pipeline (optional local load)
        if self.backend in ["transformers"]:
            try:
                from transformers import pipeline
                print(f"Loading local SLM model: {self.model_path}...")
                try:
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model_path,
                        device_map="auto"
                    )
                except Exception:
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model_path
                    )
                self.active_backend = f"Transformers ({self.model_path})"
                self.is_loaded = True
                return
            except Exception as e:
                print(f"Transformers model load fallback: {e}")

        # 3. Default to ultra-fast NPU Heuristic Simulator
        self.active_backend = "Heuristic (NPU Simulator)"
        self.is_loaded = True

    def evaluate_transcript(self, transcript_window: str) -> Dict:
        """
        Runs local inference on the rolling transcript window.
        """
        t0 = time.perf_counter()

        if not self.is_loaded:
            self.load_model()

        # Route to active backend
        if "Ollama" in self.active_backend:
            slm_verdict = self._query_ollama(transcript_window)
        elif "Transformers" in self.active_backend and self.pipeline:
            slm_verdict = self._query_transformers(transcript_window)
        else:
            slm_verdict = self._mock_npu_inference(transcript_window)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        slm_verdict["slm_latency_ms"] = elapsed_ms
        slm_verdict["executed_on_npu"] = self.use_npu
        slm_verdict["backend"] = self.active_backend
        return slm_verdict

    def _query_ollama(self, transcript: str) -> Dict:
        """Query local Ollama LLM endpoint."""
        prompt = f"{self.system_prompt}\n\nTranscript:\n\"{transcript}\"\n\nJSON Output:"
        payload = json.dumps({
            "model": "qwen2.5:1.5b",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response_text = data.get("response", "")
                parsed = json.loads(response_text)
                return {
                    "is_scam": bool(parsed.get("is_scam", True)),
                    "confidence": float(parsed.get("confidence", 0.95)),
                    "threat_type": str(parsed.get("threat_type", "SUSPICIOUS_COERCION")),
                    "intent_summary": str(parsed.get("intent_summary", "Social engineering detected.")),
                    "recommended_action": str(parsed.get("recommended_action", "DISCONNECT_AND_MUTE")),
                }
        except Exception:
            return self._mock_npu_inference(transcript)

    def _query_transformers(self, transcript: str) -> Dict:
        """Query HuggingFace Transformers pipeline."""
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\nAnalyze this call transcript:\n\"{transcript}\"<|im_end|>\n<|im_start|>assistant\n```json\n"
        try:
            outputs = self.pipeline(prompt, max_new_tokens=150, temperature=0.1)
            generated_text = outputs[0]["generated_text"]
            json_str = generated_text.split("```json")[-1].split("```")[0].strip()
            parsed = json.loads(json_str)
            return {
                "is_scam": bool(parsed.get("is_scam", True)),
                "confidence": float(parsed.get("confidence", 0.95)),
                "threat_type": str(parsed.get("threat_type", "SUSPICIOUS_COERCION")),
                "intent_summary": str(parsed.get("intent_summary", "Social engineering detected.")),
                "recommended_action": str(parsed.get("recommended_action", "DISCONNECT_AND_MUTE")),
            }
        except Exception:
            return self._mock_npu_inference(transcript)

    def _mock_npu_inference(self, text: str) -> Dict:
        """Simulates quantized SLM zero-shot token generation (<120ms TTFT)."""
        text_lower = text.lower()
        if "digital arrest" in text_lower or "police" in text_lower or "warrant" in text_lower or "cbi" in text_lower or "fedex" in text_lower:
            return {
                "is_scam": True,
                "confidence": 0.98,
                "threat_type": "DIGITAL_ARREST_COERCION",
                "intent_summary": "Caller is impersonating police/cyber crime authorities to intimidate victim with fake warrant and demand money transfer.",
                "recommended_action": "DISCONNECT_AND_MUTE"
            }
        elif "upi pin" in text_lower or "collect request" in text_lower or "phonepe" in text_lower or "gpay" in text_lower:
            return {
                "is_scam": True,
                "confidence": 0.96,
                "threat_type": "UPI_EXTRACTION_FRAUD",
                "intent_summary": "Caller is attempting to trick user into entering UPI PIN under the false premise of receiving money.",
                "recommended_action": "BLOCK_PAYMENT_APP"
            }
        elif "sim" in text_lower or "kyc" in text_lower or "anydesk" in text_lower or "trai" in text_lower:
            return {
                "is_scam": True,
                "confidence": 0.95,
                "threat_type": "FAKE_KYC_REMOTE_ACCESS",
                "intent_summary": "Caller is threatening SIM deactivation to force remote access APK download and credential theft.",
                "recommended_action": "BLOCK_CALL_AND_REVOKE_PERMISSIONS"
            }
        return {
            "is_scam": False,
            "confidence": 0.85,
            "threat_type": "BENIGN_OR_TELEMARKETING",
            "intent_summary": "No active legal or financial coercion detected.",
            "recommended_action": "ALLOW_CALL"
        }
