"""
Audio Stream Chunker & Streaming ASR Manager
Handles live 16kHz mono audio streams with rolling overlap, VAD gating,
and transcript deduplication for real-time scam scoring.
"""

import time
import numpy as np
from typing import Callable, Dict, List, Optional
from .streaming import StreamingScamDetector
from .tier2_slm_judge import Tier2SLMJudge
from .interception_hud import InterceptionHUD


class AudioStreamChunker:
    """
    Slices raw PCM audio streams into overlapping windows for low-latency ASR inference.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_sec: float = 0.5,
        window_duration_sec: float = 2.0,
        energy_threshold: float = 0.012,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_sec)      # 8,000 samples (500ms)
        self.window_size = int(sample_rate * window_duration_sec)    # 32,000 samples (2000ms)
        self.energy_threshold = energy_threshold

        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.confirmed_transcript_tokens: List[str] = []

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Lightweight Root Mean Square (RMS) energy gate to filter dead silence."""
        if len(audio_chunk) == 0:
            return False
        rms = np.sqrt(np.mean(np.square(audio_chunk)))
        return bool(rms >= self.energy_threshold)

    def add_audio_frame(self, pcm_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Ingests a 500ms PCM chunk (float32 normalized between -1.0 and 1.0).
        Returns a 2.0s window for ASR if speech is present; otherwise returns None.
        """
        # Append incoming chunk to circular ring buffer
        self.audio_buffer = np.concatenate((self.audio_buffer, pcm_data))

        # Maintain maximum buffer length
        if len(self.audio_buffer) > self.window_size:
            self.audio_buffer = self.audio_buffer[-self.window_size:]

        # Require a full context window before emitting for ASR
        if len(self.audio_buffer) < self.window_size:
            return None

        # Check speech presence across the active window
        if not self.is_speech(self.audio_buffer[-self.chunk_size:]):
            return None

        return self.audio_buffer.copy()

    def deduplicate_transcript(self, window_text: str) -> str:
        """
        Extracts new words from overlapping ASR context windows using prefix alignment.
        """
        new_tokens = window_text.strip().split()
        if not new_tokens:
            return ""

        if not self.confirmed_transcript_tokens:
            self.confirmed_transcript_tokens = new_tokens
            return " ".join(new_tokens)

        # Find maximum token overlap between confirmed tokens and current window
        max_overlap = min(len(self.confirmed_transcript_tokens), len(new_tokens))
        overlap_idx = 0

        for k in range(max_overlap, 0, -1):
            if self.confirmed_transcript_tokens[-k:] == new_tokens[:k]:
                overlap_idx = k
                break

        emitted_tokens = new_tokens[overlap_idx:]
        self.confirmed_transcript_tokens.extend(emitted_tokens)
        return " ".join(emitted_tokens)

    def reset(self):
        """Resets the audio ring buffer and transcript cache."""
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.confirmed_transcript_tokens = []


class LiveCallAudioInterceptor:
    """
    Live Stream Orchestrator
    Streams raw PCM chunks -> AudioStreamChunker -> ASR Engine -> StreamingScamDetector -> Tier2SLMJudge -> InterceptionHUD

    Tier 2 Debounce Policy:
        The SLM judge is expensive (~100ms+ with real weights).  To avoid a
        re-trigger storm where every post-breach chunk invokes the judge, we
        latch after the first invocation and only re-invoke when:
          • A new threat category appears that wasn't in the initial verdict.
          • The score crosses a higher severity band (e.g., HIGH -> CRITICAL).
    """

    # Severity bands used for re-trigger gating
    _SEVERITY_BANDS = [(75.0, "CRITICAL"), (50.0, "HIGH"), (25.0, "MEDIUM"), (0.0, "LOW")]

    def __init__(
        self,
        asr_engine: Callable[[np.ndarray], str],
        telemetry_callback: Optional[Callable[[Dict], None]] = None,
        use_npu: bool = False
    ):
        """
        Args:
            asr_engine: Callable accepting 1D float32 audio array (16kHz) and returning text.
            telemetry_callback: Optional callback for telemetry HUD dispatch.
            use_npu: Whether Tier 2 SLM judge targets Snapdragon NPU execution provider.
        """
        self.chunker = AudioStreamChunker()
        self.asr_engine = asr_engine
        self.detector = StreamingScamDetector(window_size_words=150)
        self.slm_judge = Tier2SLMJudge(use_npu=use_npu)
        self.hud = InterceptionHUD(telemetry_callback=telemetry_callback)

        # ── Tier 2 debounce state ───────────────────────────────────
        self._tier2_triggered: bool = False
        self._tier2_cached_verdict: Optional[Dict] = None
        self._tier2_seen_categories: set = set()
        self._tier2_severity_band: str = "LOW"

    def _get_severity_band(self, score: float) -> str:
        """Map a score to its severity band."""
        for threshold, band in self._SEVERITY_BANDS:
            if score >= threshold:
                return band
        return "LOW"

    def _should_retrigger_tier2(self, result: Dict) -> bool:
        """
        Determine whether a new Tier 2 invocation is warranted.

        Returns True if:
          1. Tier 2 has never fired (first breach).
          2. New threat categories have appeared since the last invocation.
          3. The effective score has crossed into a higher severity band.
        """
        if not self._tier2_triggered:
            return True

        # Check for new categories not seen by the previous Tier 2 run
        current_cats = set(result.get("categories_triggered", []))
        new_cats = current_cats - self._tier2_seen_categories
        if new_cats:
            return True

        # Check for severity band escalation
        effective_threat = max(result["cumulative_score"], result["peak_score"])
        current_band = self._get_severity_band(effective_threat)
        band_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        if band_order.get(current_band, 0) > band_order.get(self._tier2_severity_band, 0):
            return True

        return False

    def on_audio_chunk_received(self, pcm_chunk_500ms: np.ndarray) -> Optional[Dict]:
        """
        Processes a live 500ms audio packet. Returns scam analysis if new speech is transcribed.
        Triggers Tier 2 Edge SLM Judge and HUD alert when risk threshold is crossed.
        """
        t0 = time.perf_counter()

        # Step 1: Ingest audio into ring buffer & check VAD
        audio_window = self.chunker.add_audio_frame(pcm_chunk_500ms)
        if audio_window is None:
            return None

        # Step 2: Run local streaming ASR inference
        raw_window_transcript = self.asr_engine(audio_window)

        # Step 3: Deduplicate overlapping context tokens
        new_transcript_delta = self.chunker.deduplicate_transcript(raw_window_transcript)
        if not new_transcript_delta:
            return None

        # Step 4: Stream transcript delta into Layer 1 ensemble detector
        result = self.detector.process_chunk(new_transcript_delta)
        result["transcript_delta"] = new_transcript_delta

        # Step 5: Tier 2 Edge SLM Judge Integration (with debounce latch)
        # Use max(current_score, peak_score) to prevent gate trigger starvation.
        tier2_verdict = None
        effective_threat = max(result["cumulative_score"], result["peak_score"])

        if effective_threat >= 60.0 or result.get("needs_l2_review"):
            if self._should_retrigger_tier2(result):
                # Full SLM invocation — pass the complete transcript so the
                # judge always has full context, not the decayed sliding window.
                tier2_verdict = self.slm_judge.evaluate_transcript(
                    self.detector.full_transcript
                )

                # Update debounce state
                self._tier2_triggered = True
                self._tier2_cached_verdict = tier2_verdict
                self._tier2_seen_categories |= set(
                    result.get("categories_triggered", [])
                )
                self._tier2_severity_band = self._get_severity_band(effective_threat)
                result["tier2_source"] = "invoked"
            else:
                # Reuse the cached verdict — no SLM call, zero extra latency.
                tier2_verdict = self._tier2_cached_verdict
                result["tier2_source"] = "cached"

            result["tier2_slm"] = tier2_verdict

        # Step 6: HUD Alert Trigger
        self.hud.trigger_alert(result, tier2_verdict)

        # Capture total E2E latency AFTER all steps (including Tier 2 SLM)
        result["total_e2e_latency_ms"] = (time.perf_counter() - t0) * 1000

        return result
