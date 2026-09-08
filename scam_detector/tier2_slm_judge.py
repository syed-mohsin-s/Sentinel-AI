"""
Tier 2 Edge SLM Judge
Executes local SLMs (Gemma 3 1B-IT / Qwen2.5 / Llama 3.2) via LiteRT-LM,
HuggingFace Transformers, Ollama, llama-cpp-python (GGUF), ONNX Runtime GenAI,
or fast NPU heuristic fallback.
Generates structured JSON coercion reasoning.

Backend resolution order (when backend="auto"):
    1. llama-cpp-python — loads a quantized GGUF file (on-device, preferred)
    2. ONNX Runtime GenAI — loads an ONNX-exported model directory
    3. LiteRT-LM        — Google edge runtime (Gemma3-1B-IT .litertlm)
    4. Ollama            — queries localhost:11434
    5. Transformers      — loads a full HuggingFace model
    6. Heuristic mock    — always-available keyword fallback
"""

import json
import os
import time
import urllib.request
from typing import Dict, Optional


class Tier2SLMJudge:
    """
    Tier 2 Edge SLM Reasoning Engine.
    Supports LiteRT-LM (Gemma3-1B-IT), HuggingFace Transformers pipeline,
    Ollama API, llama-cpp-python GGUF, ONNX Runtime GenAI, and fast NPU
    heuristic fallback.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_npu: bool = False,
        backend: str = "auto"  # Options: "auto", "litert", "transformers", "ollama", "llama_cpp", "onnx_genai", "heuristic"
    ):
        self.model_path = model_path or "litert-community/Gemma3-1B-IT"
        self.use_npu = use_npu
        self.backend = backend
        self.pipeline = None
        self._llama_model = None
        self._onnx_model = None
        self._onnx_tokenizer = None
        self._litert_engine = None
        self.is_loaded = False
        self.active_backend = "Heuristic (NPU Simulator)"
        self._last_used_npu = False  # Tracks what actually happened, not what was requested
        self._last_backend = "none"

        self.system_prompt = (
            "You are Sentinel AI, an edge-native real-time fraud judge. "
            "Analyze the call transcript for social engineering, coercion, or authority impersonation. "
            "Output strict JSON only with keys: 'is_scam' (bool), 'confidence' (float 0-1), "
            "'threat_type' (str), 'intent_summary' (str), 'recommended_action' (str)."
        )

        # GBNF grammar for structured JSON output from small models.
        # Constrains autoregressive decoding to produce only valid JSON
        # matching the exact schema we expect, preventing malformed output
        # from quantized 0.5B models (unterminated quotes, missing braces, etc).
        self._json_gbnf_grammar = r'''
root   ::= "{" ws members ws "}"
members ::= pair ("," ws pair)*
pair   ::= ws string ws ":" ws value
value  ::= string | number | boolean | "null"
string ::= "\"" chars "\""
chars  ::= char*
char   ::= [^"\\] | "\\" escape
escape ::= ["\\bfnrt/]
number ::= "-"? digits ("." digits)?
digits ::= [0-9]+
boolean::= "true" | "false"
ws     ::= [ \t\n]*
'''

    # ─── Model Loading ──────────────────────────────────────────────

    def load_model(self):
        """Attempts to load a live local LLM backend if available."""
        if self.is_loaded:
            return

        # 1. llama-cpp-python GGUF (preferred — on-device, no external daemon)
        if self.backend in ["auto", "llama_cpp"]:
            if self._try_load_llama_cpp():
                return

        # 2. ONNX Runtime GenAI
        if self.backend in ["auto", "onnx_genai"]:
            if self._try_load_onnx_genai():
                return

        # 3. LiteRT-LM (Google edge runtime — Gemma3-1B-IT)
        if self.backend in ["auto", "litert"]:
            if self._try_load_litert():
                return

        # 4. Ollama API endpoint (external daemon — may hijack inference if
        #    an unrelated Ollama instance is running in the background)
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

        # 5. HuggingFace Transformers (heavy, slow on CPU)
        if self.backend in ["auto", "transformers"]:
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

        # 6. Default — ultra-fast NPU Heuristic Simulator (always available)
        self.active_backend = "Heuristic (NPU Simulator)"
        self.is_loaded = True

    # ─── Backend Loaders ────────────────────────────────────────────

    def _try_load_litert(self) -> bool:
        """Attempt to load a Gemma3-1B-IT model via LiteRT-LM (Google edge runtime)."""
        try:
            import litert_lm
        except ImportError:
            return False

        litert_path = os.environ.get("SENTINEL_LITERT_PATH", "")
        if not litert_path:
            # Check if model_path points to a .litertlm file
            if self.model_path.endswith(".litertlm") and os.path.isfile(self.model_path):
                litert_path = self.model_path
            else:
                # Default: check models/ directory
                default_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models", "gemma3-1b-it.litertlm"
                )
                if os.path.isfile(default_path):
                    litert_path = default_path

        if not litert_path or not os.path.isfile(litert_path):
            return False

        try:
            # Select hardware backend: NPU/GPU if requested, CPU otherwise
            if self.use_npu:
                try:
                    backend = litert_lm.Backend.NPU()
                except Exception:
                    try:
                        backend = litert_lm.Backend.GPU()
                    except Exception:
                        backend = litert_lm.Backend.CPU()
            else:
                backend = litert_lm.Backend.CPU()

            self._litert_engine = litert_lm.Engine(litert_path, backend=backend)
            self.active_backend = f"LiteRT-LM (Gemma3-1B-IT, {os.path.basename(litert_path)})"
            self.is_loaded = True
            print(f"LiteRT-LM engine loaded: {litert_path}")
            return True
        except Exception as e:
            print(f"LiteRT-LM load failed: {e}")
            self._litert_engine = None
            return False

    def _try_load_llama_cpp(self) -> bool:
        """Attempt to load a GGUF model via llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            return False

        gguf_path = os.environ.get("SENTINEL_GGUF_PATH", "")
        if not gguf_path:
            # Fallback: check if model_path points to a .gguf file
            if self.model_path.endswith(".gguf") and os.path.isfile(self.model_path):
                gguf_path = self.model_path

        if not gguf_path or not os.path.isfile(gguf_path):
            return False

        try:
            self._llama_model = Llama(
                model_path=gguf_path,
                n_ctx=1024,
                n_threads=4,
                n_gpu_layers=0,  # CPU-only by default; set via env if GPU available
                verbose=False,
            )
            self.active_backend = f"llama-cpp-python ({os.path.basename(gguf_path)})"
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"llama-cpp-python load failed: {e}")
            return False

    def _try_load_onnx_genai(self) -> bool:
        """Attempt to load an ONNX model directory via onnxruntime-genai."""
        try:
            import onnxruntime_genai as og
        except ImportError:
            return False

        onnx_dir = os.environ.get("SENTINEL_ONNX_PATH", "")
        if not onnx_dir:
            # Fallback: check if model_path is a directory containing ONNX files
            if os.path.isdir(self.model_path):
                onnx_dir = self.model_path

        if not onnx_dir or not os.path.isdir(onnx_dir):
            return False

        try:
            self._onnx_model = og.Model(onnx_dir)
            self._onnx_tokenizer = og.Tokenizer(self._onnx_model)
            self.active_backend = f"ONNX Runtime GenAI ({os.path.basename(onnx_dir)})"
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"ONNX Runtime GenAI load failed: {e}")
            return False

    # ─── Inference Dispatch ─────────────────────────────────────────

    def evaluate_transcript(self, transcript_window: str) -> Dict:
        """
        Runs local inference on the rolling transcript window.
        """
        t0 = time.perf_counter()

        if not self.is_loaded:
            self.load_model()

        # Route to active backend
        if "LiteRT" in self.active_backend:
            slm_verdict = self._query_litert(transcript_window)
            # LiteRT-LM: NPU if Backend.NPU() was selected during load
            self._last_used_npu = self.use_npu and self._litert_engine is not None
            self._last_backend = "litert"
        elif "Ollama" in self.active_backend:
            slm_verdict = self._query_ollama(transcript_window)
            self._last_used_npu = False  # Ollama does not configure an NPU provider
            self._last_backend = "ollama"
        elif "llama-cpp" in self.active_backend:
            slm_verdict = self._query_llama_cpp(transcript_window)
            # llama-cpp-python: NPU only if real QNN offload is configured.
            # Currently n_gpu_layers=0 (CPU-only), so always False.
            self._last_used_npu = False
            self._last_backend = "llama_cpp"
        elif "ONNX" in self.active_backend:
            slm_verdict = self._query_onnx_genai(transcript_window)
            # ONNX GenAI: check if a QNN/NPU execution provider was actually used
            self._last_used_npu = self._check_onnx_npu_provider()
            self._last_backend = "onnx_genai"
        elif "Transformers" in self.active_backend and self.pipeline:
            slm_verdict = self._query_transformers(transcript_window)
            self._last_used_npu = False  # Transformers does not configure an NPU provider
            self._last_backend = "transformers"
        else:
            slm_verdict = self._mock_npu_inference(transcript_window)
            self._last_used_npu = False  # Heuristic mock, no real NPU
            self._last_backend = "heuristic"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        slm_verdict["slm_latency_ms"] = elapsed_ms
        slm_verdict["executed_on_npu"] = self._last_used_npu
        slm_verdict["backend"] = self.active_backend
        return slm_verdict

    # ─── Backend Query Methods ──────────────────────────────────────

    def _query_litert(self, transcript: str) -> Dict:
        """Run inference via LiteRT-LM Engine (Gemma3-1B-IT on-device).

        Creates a conversation session, sends the transcript, and collects
        the streamed response chunks into a complete JSON verdict.
        """
        import litert_lm

        prompt = f'Analyze this call transcript:\n"{transcript}"'

        try:
            messages = [litert_lm.Message.system(self.system_prompt)]

            with self._litert_engine.create_conversation(messages=messages) as conversation:
                # Collect streaming chunks into full response
                full_response = ""
                for chunk in conversation.send_message_async(prompt):
                    text = chunk.get("content", [{}])[0].get("text", "")
                    full_response += text

            # Extract JSON from response
            json_str = full_response
            if "```json" in json_str:
                json_str = json_str.split("```json")[-1].split("```")[0].strip()
            elif "{" in json_str:
                start = json_str.index("{")
                depth, end = 0, start
                for i in range(start, len(json_str)):
                    if json_str[i] == "{":
                        depth += 1
                    elif json_str[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                json_str = json_str[start:end]

            parsed = json.loads(json_str)
            return {
                "is_scam": bool(parsed.get("is_scam", True)),
                "confidence": float(parsed.get("confidence", 0.95)),
                "threat_type": str(parsed.get("threat_type", "SUSPICIOUS_COERCION")),
                "intent_summary": str(parsed.get("intent_summary", "Social engineering detected.")),
                "recommended_action": str(parsed.get("recommended_action", "DISCONNECT_AND_MUTE")),
            }
        except Exception as e:
            print(f"LiteRT-LM inference fallback ({e})")
            return self._mock_npu_inference(transcript)

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

    def _query_llama_cpp(self, transcript: str) -> Dict:
        """Run autoregressive decoding via llama-cpp-python on a GGUF model.

        Uses a strict GBNF grammar to constrain the 0.5B model's output to
        valid JSON, preventing the frequent malformed-output failures that
        small quantized models produce (unterminated quotes, missing braces,
        conversational preambles like 'Here is the JSON:').
        """
        try:
            from llama_cpp import LlamaGrammar
            grammar = LlamaGrammar.from_string(self._json_gbnf_grammar)
        except Exception:
            grammar = None

        try:
            # Build chat completion kwargs; use grammar if available,
            # otherwise fall back to response_format hint.
            chat_kwargs = dict(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f'Analyze this call transcript:\n"{transcript}"'},
                ],
                max_tokens=256,
                temperature=0.1,
            )
            if grammar is not None:
                chat_kwargs["grammar"] = grammar
            else:
                chat_kwargs["response_format"] = {"type": "json_object"}

            output = self._llama_model.create_chat_completion(**chat_kwargs)
            response_text = output["choices"][0]["message"]["content"]
            parsed = json.loads(response_text)
            return {
                "is_scam": bool(parsed.get("is_scam", True)),
                "confidence": float(parsed.get("confidence", 0.95)),
                "threat_type": str(parsed.get("threat_type", "SUSPICIOUS_COERCION")),
                "intent_summary": str(parsed.get("intent_summary", "Social engineering detected.")),
                "recommended_action": str(parsed.get("recommended_action", "DISCONNECT_AND_MUTE")),
            }
        except Exception as e:
            print(f"llama-cpp-python inference fallback ({e})")
            return self._mock_npu_inference(transcript)

    def _query_onnx_genai(self, transcript: str) -> Dict:
        """Run autoregressive decoding via ONNX Runtime GenAI."""
        import onnxruntime_genai as og

        prompt = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nAnalyze this call transcript:\n\"{transcript}\"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            input_tokens = self._onnx_tokenizer.encode(prompt)

            params = og.GeneratorParams(self._onnx_model)
            params.set_search_options(max_length=256, temperature=0.1)
            params.input_ids = input_tokens

            generator = og.Generator(self._onnx_model, params)
            output_tokens = []
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                output_tokens.append(new_token)

            response_text = self._onnx_tokenizer.decode(output_tokens)

            # Extract JSON from response
            json_str = response_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[-1].split("```")[0].strip()
            elif "{" in json_str:
                # Find the first complete JSON object
                start = json_str.index("{")
                depth, end = 0, start
                for i in range(start, len(json_str)):
                    if json_str[i] == "{":
                        depth += 1
                    elif json_str[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                json_str = json_str[start:end]

            parsed = json.loads(json_str)
            return {
                "is_scam": bool(parsed.get("is_scam", True)),
                "confidence": float(parsed.get("confidence", 0.95)),
                "threat_type": str(parsed.get("threat_type", "SUSPICIOUS_COERCION")),
                "intent_summary": str(parsed.get("intent_summary", "Social engineering detected.")),
                "recommended_action": str(parsed.get("recommended_action", "DISCONNECT_AND_MUTE")),
            }
        except Exception as e:
            print(f"ONNX Runtime GenAI inference fallback ({e})")
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

    def _check_onnx_npu_provider(self) -> bool:
        """Check whether the ONNX Runtime session is actually using a QNN/NPU execution provider."""
        try:
            import onnxruntime
            # If model exposes a session, check its providers
            if hasattr(self._onnx_model, 'session') and hasattr(self._onnx_model.session, 'get_providers'):
                providers = self._onnx_model.session.get_providers()
                npu_providers = {"QNNExecutionProvider", "DmlExecutionProvider"}
                return bool(set(providers) & npu_providers)
        except Exception:
            pass
        # Cannot confirm NPU provider — default to False (honest reporting)
        return False
