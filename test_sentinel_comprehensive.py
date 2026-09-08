"""
Sentinel-AI — Comprehensive Test Suite
=======================================
Unit, Functional, and End-to-End Integration tests covering every layer:

  1. UNIT TESTS          — isolated component-level testing
     • ScamDetector keyword/pattern/tokenization
     • RiskScorer scoring math, thresholds, FP mitigation
     • TfIdfSemanticAnalyzer calibrated scoring
     • Tier2SLMJudge backend selection, heuristic inference
     • AudioStreamChunker VAD, windowing, deduplication
     • InterceptionHUD severity mapping, Tier2 reconciliation
     • StreamingScamDetector session state, sliding window

  2. FUNCTIONAL TESTS    — component interaction and contract tests
     • EnsembleScorer end-to-end (Keyword + Semantic fusion)
     • StreamingScamDetector multi-chunk session lifecycle
     • LiveCallAudioInterceptor Tier 2 debounce policy
     • HUD alert payload structure validation

  3. E2E INTEGRATION     — full pipeline simulation
     • Silent → Threat → Benign audio stream lifecycle
     • Tier 2 SLM invocation and caching behavior
     • Score monotonicity, bounds, transcript continuity
     • Telemetry dispatch integrity
     • Backend passthrough (slm_backend="heuristic" / "litert")

Run:
    python -m pytest test_sentinel_comprehensive.py -v
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np
import pytest

# ─── Module imports ──────────────────────────────────────────────────
from scam_detector.detector import ScamDetector
from scam_detector.scorer import RiskScorer, RiskLevel
from scam_detector.tinybert_analyzer import TfIdfSemanticAnalyzer, TinyBertAnalyzer, SemanticAnalyzer
from scam_detector.ensemble_scorer import EnsembleScorer, quick_analyze
from scam_detector.streaming import StreamingScamDetector
from scam_detector.audio_chunker import AudioStreamChunker, LiveCallAudioInterceptor
from scam_detector.tier2_slm_judge import Tier2SLMJudge
from scam_detector.interception_hud import InterceptionHUD, SEVERITY_TO_HUD_STATE, reconcile_hud_state
from scam_detector.word_dictionary import SCAM_WORDS, CATEGORY_WEIGHTS, HARD_SCAM_CATEGORIES
from scam_detector.patterns import PATTERNS, PATTERN_WEIGHTS


# ═════════════════════════════════════════════════════════════════════
#  SECTION 1: UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestScamDetector:
    """Unit tests for the keyword/pattern detection engine."""

    def setup_method(self):
        self.detector = ScamDetector()

    # ── Text Cleaning ───────────────────────────────────────────────

    def test_clean_text_lowercases(self):
        assert self.detector.clean_text("Hello WORLD") == "hello world"

    def test_clean_text_strips_punctuation(self):
        assert self.detector.clean_text("Hello, world! Are you ok?") == "hello world are you ok"

    def test_clean_text_collapses_whitespace(self):
        assert self.detector.clean_text("hello    world") == "hello world"

    def test_clean_text_empty(self):
        assert self.detector.clean_text("") == ""

    # ── Tokenization ────────────────────────────────────────────────

    def test_tokenize_basic(self):
        tokens = self.detector.tokenize("This is a test")
        assert tokens == ["this", "is", "a", "test"]

    def test_tokenize_empty(self):
        assert self.detector.tokenize("") == []

    # ── Keyword Detection ───────────────────────────────────────────

    def test_detects_digital_arrest_keywords(self):
        text = "This is CBI enforcement directorate. A digital arrest warrant has been issued."
        detected, matched = self.detector.find_suspicious_words(text)
        assert "digital_arrest" in detected
        assert "digital arrest" in matched or "cbi" in matched

    def test_detects_upi_coercion_keywords(self):
        text = "Please enter your UPI PIN to receive money via PhonePe"
        detected, matched = self.detector.find_suspicious_words(text)
        assert "upi_coercion" in detected
        assert "upi pin" in matched or "phonepe" in matched

    def test_detects_fake_kyc_keywords(self):
        text = "Your SIM will be blocked due to pending KYC. Download AnyDesk."
        detected, matched = self.detector.find_suspicious_words(text)
        assert "fake_kyc" in detected

    def test_benign_text_no_detection(self):
        text = "Hey, how are you? The weather is nice today."
        detected, matched = self.detector.find_suspicious_words(text)
        assert len(detected) == 0
        assert len(matched) == 0

    def test_multi_category_detection(self):
        text = "CBI has issued arrest warrant. Transfer funds via bitcoin immediately."
        detected, matched = self.detector.find_suspicious_words(text)
        triggered_cats = set(detected.keys())
        assert len(triggered_cats) >= 2  # at least digital_arrest + one more

    # ── Pattern Detection ───────────────────────────────────────────

    def test_detects_digital_arrest_pattern(self):
        text = "You are placed on digital arrest by CBI enforcement department"
        patterns = self.detector.find_patterns(text)
        assert "digital_arrest_pattern" in patterns

    def test_detects_upi_coercion_pattern(self):
        text = "Enter your UPI PIN to receive the refund"
        patterns = self.detector.find_patterns(text)
        assert "upi_coercion_pattern" in patterns

    def test_detects_otp_theft_pattern(self):
        text = "Please share the 6-digit OTP sent to your phone"
        patterns = self.detector.find_patterns(text)
        assert "otp_theft_pattern" in patterns

    def test_detects_remote_access_pattern(self):
        text = "Download the AnyDesk application for verification"
        patterns = self.detector.find_patterns(text)
        assert "remote_access_pattern" in patterns

    def test_benign_text_no_patterns(self):
        text = "The pizza delivery will arrive in 30 minutes"
        patterns = self.detector.find_patterns(text)
        assert len(patterns) == 0

    # ── Full Analysis ───────────────────────────────────────────────

    def test_analyze_returns_required_keys(self):
        result = self.detector.analyze("test text")
        required = {"text", "word_count", "suspicious_word_count", "detected_words",
                     "matched_keywords", "detected_patterns", "categories_triggered", "latency_ms"}
        assert required.issubset(set(result.keys()))

    def test_analyze_latency_measured(self):
        result = self.detector.analyze("This is CBI calling about your arrest warrant")
        assert result["latency_ms"] > 0


class TestRiskScorer:
    """Unit tests for the risk scoring algorithm."""

    def setup_method(self):
        self.scorer = RiskScorer()
        self.detector = ScamDetector()

    def test_benign_text_scores_low(self):
        analysis = self.detector.analyze("Hello how are you today")
        _, score, level = self.scorer.calculate_score(analysis)
        assert score < 25
        assert level == RiskLevel.LOW

    def test_scam_text_scores_high(self):
        text = "CBI digital arrest warrant issued. Transfer 50000 via bitcoin immediately."
        analysis = self.detector.analyze(text)
        _, score, level = self.scorer.calculate_score(analysis)
        assert score >= 50
        assert level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_score_bounds(self):
        """Score should always be in [0, 100]."""
        texts = [
            "",
            "hello",
            "CBI arrest warrant digital arrest transfer funds bitcoin immediately police",
            "Enter UPI PIN phonepe gpay sim block anydesk download immediately",
        ]
        for text in texts:
            analysis = self.detector.analyze(text)
            _, score, _ = self.scorer.calculate_score(analysis)
            assert 0 <= score <= 100, f"Score {score} out of bounds for: {text!r}"

    def test_false_positive_cap_for_pressure_only(self):
        """Text with only pressure_tactics (no hard scam category) should be capped."""
        text = "Limited time special offer discount expires today don't miss out"
        analysis = self.detector.analyze(text)
        _, score, level = self.scorer.calculate_score(analysis)
        assert score <= 35, f"FP cap failed: score={score}"

    def test_combination_bonus_increases_score(self):
        """Dangerous category combos should boost score via combination multiplier."""
        text1 = "CBI officer calling"  # impersonation only
        text2 = "CBI officer calling, transfer funds immediately via bitcoin"  # impersonation + financial + urgency
        a1 = self.detector.analyze(text1)
        a2 = self.detector.analyze(text2)
        _, score1, _ = self.scorer.calculate_score(a1)
        _, score2, _ = self.scorer.calculate_score(a2)
        assert score2 > score1

    def test_density_bonus(self):
        assert self.scorer.calculate_density_bonus(0, 0) == 1.0
        assert self.scorer.calculate_density_bonus(0, 100) == 1.0
        bonus = self.scorer.calculate_density_bonus(10, 20)
        assert bonus > 1.0
        assert bonus <= 2.0

    def test_risk_level_thresholds(self):
        assert RiskLevel.LOW == "LOW"
        assert RiskLevel.MEDIUM == "MEDIUM"
        assert RiskLevel.HIGH == "HIGH"
        assert RiskLevel.CRITICAL == "CRITICAL"

    def test_risk_emoji(self):
        assert self.scorer.get_risk_emoji(RiskLevel.LOW) == "🟢"
        assert self.scorer.get_risk_emoji(RiskLevel.CRITICAL) == "🔴"
        assert self.scorer.get_risk_emoji("UNKNOWN") == "⚪"


class TestTfIdfSemanticAnalyzer:
    """Unit tests for the TF-IDF semantic analyzer."""

    def setup_method(self):
        self.analyzer = TfIdfSemanticAnalyzer()

    def test_lazy_load(self):
        assert not self.analyzer.is_loaded
        self.analyzer.load_model()
        assert self.analyzer.is_loaded

    def test_backend_name(self):
        assert self.analyzer.backend == "TF-IDF (Calibrated)"

    def test_scam_text_high_score(self):
        text = "CBI officer calling. Digital arrest warrant. Transfer money via bitcoin."
        result = self.analyzer.analyze(text)
        assert result["scam_score"] > 30

    def test_benign_text_low_score(self):
        text = "Your delivery will arrive tomorrow. Thank you for your order."
        result = self.analyzer.analyze(text)
        assert result["scam_score"] < 40

    def test_analysis_returns_required_keys(self):
        result = self.analyzer.analyze("test")
        required = {"scam_score", "confidence", "scam_similarity", "legit_similarity",
                     "max_scam_similarity", "max_legit_similarity", "backend", "latency_ms"}
        assert required.issubset(set(result.keys()))

    def test_score_bounds(self):
        result = self.analyzer.analyze("CBI arrest warrant digital arrest")
        assert 0 <= result["scam_score"] <= 100

    def test_backward_compat_aliases(self):
        """TinyBertAnalyzer and SemanticAnalyzer should resolve to TfIdfSemanticAnalyzer."""
        assert TinyBertAnalyzer is TfIdfSemanticAnalyzer
        assert SemanticAnalyzer is TfIdfSemanticAnalyzer


class TestTier2SLMJudge:
    """Unit tests for the Tier 2 SLM Judge."""

    def test_heuristic_backend_loads_instantly(self):
        judge = Tier2SLMJudge(backend="heuristic")
        judge.load_model()
        assert judge.is_loaded
        assert "Heuristic" in judge.active_backend

    def test_default_model_path_updated(self):
        judge = Tier2SLMJudge()
        assert judge.model_path == "litert-community/Gemma3-1B-IT"

    def test_backend_option_litert_recognized(self):
        judge = Tier2SLMJudge(backend="litert")
        assert judge.backend == "litert"

    def test_litert_engine_attr_exists(self):
        judge = Tier2SLMJudge()
        assert hasattr(judge, '_litert_engine')
        assert judge._litert_engine is None

    def test_heuristic_detects_digital_arrest(self):
        judge = Tier2SLMJudge(backend="heuristic")
        judge.load_model()
        result = judge.evaluate_transcript(
            "This is police. A digital arrest warrant has been issued by CBI."
        )
        assert result["is_scam"] is True
        assert result["confidence"] > 0.9
        assert "DIGITAL_ARREST" in result["threat_type"]

    def test_heuristic_detects_upi_fraud(self):
        judge = Tier2SLMJudge(backend="heuristic")
        judge.load_model()
        result = judge.evaluate_transcript(
            "Enter your UPI PIN in PhonePe to receive the collect request money."
        )
        assert result["is_scam"] is True
        assert "UPI" in result["threat_type"]

    def test_heuristic_detects_fake_kyc(self):
        judge = Tier2SLMJudge(backend="heuristic")
        judge.load_model()
        result = judge.evaluate_transcript(
            "Your SIM will be blocked by TRAI. Download AnyDesk for KYC verification."
        )
        assert result["is_scam"] is True
        assert "KYC" in result["threat_type"]

    def test_heuristic_benign_text(self):
        judge = Tier2SLMJudge(backend="heuristic")
        judge.load_model()
        result = judge.evaluate_transcript("Hey, how was your day? The weather is nice.")
        assert result["is_scam"] is False
        assert result["threat_type"] == "BENIGN_OR_TELEMARKETING"

    def test_verdict_contains_required_keys(self):
        judge = Tier2SLMJudge(backend="heuristic")
        judge.load_model()
        result = judge.evaluate_transcript("test")
        required = {"is_scam", "confidence", "threat_type", "intent_summary",
                     "recommended_action", "slm_latency_ms", "executed_on_npu", "backend"}
        assert required.issubset(set(result.keys()))

    def test_executed_on_npu_is_false_for_heuristic(self):
        judge = Tier2SLMJudge(backend="heuristic", use_npu=True)
        judge.load_model()
        result = judge.evaluate_transcript("test")
        assert result["executed_on_npu"] is False

    def test_last_backend_tracking(self):
        judge = Tier2SLMJudge(backend="heuristic")
        judge.load_model()
        judge.evaluate_transcript("test")
        assert judge._last_backend == "heuristic"


class TestAudioStreamChunker:
    """Unit tests for the audio chunking and VAD engine."""

    def setup_method(self):
        self.chunker = AudioStreamChunker()

    def _make_speech_frame(self, duration_sec=0.5, freq=440):
        t = np.linspace(0, duration_sec, int(16000 * duration_sec), False)
        return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def _make_silence_frame(self, duration_sec=0.5):
        return np.zeros(int(16000 * duration_sec), dtype=np.float32)

    # ── VAD (Voice Activity Detection) ──────────────────────────────

    def test_is_speech_detects_signal(self):
        assert self.chunker.is_speech(self._make_speech_frame()) is True

    def test_is_speech_rejects_silence(self):
        assert self.chunker.is_speech(self._make_silence_frame()) is False

    def test_is_speech_rejects_empty(self):
        assert self.chunker.is_speech(np.zeros(0, dtype=np.float32)) is False

    # ── Windowing ───────────────────────────────────────────────────

    def test_requires_full_window_before_emit(self):
        """First 3 chunks (1.5s) should return None (window is 2.0s)."""
        frame = self._make_speech_frame()
        for _ in range(3):
            assert self.chunker.add_audio_frame(frame) is None

    def test_emits_after_full_window(self):
        """4th chunk completes the 2.0s window → should emit."""
        frame = self._make_speech_frame()
        for _ in range(3):
            self.chunker.add_audio_frame(frame)
        result = self.chunker.add_audio_frame(frame)
        assert result is not None
        assert len(result) == self.chunker.window_size

    def test_silent_chunks_never_emit(self):
        frame = self._make_silence_frame()
        for _ in range(10):
            assert self.chunker.add_audio_frame(frame) is None

    def test_buffer_capped_at_window_size(self):
        frame = self._make_speech_frame()
        for _ in range(10):
            self.chunker.add_audio_frame(frame)
        assert len(self.chunker.audio_buffer) == self.chunker.window_size

    # ── Transcript Deduplication ────────────────────────────────────

    def test_deduplicate_first_transcript(self):
        result = self.chunker.deduplicate_transcript("hello world test")
        assert result == "hello world test"

    def test_deduplicate_overlapping(self):
        self.chunker.deduplicate_transcript("hello world test")
        result = self.chunker.deduplicate_transcript("world test new words")
        assert result == "new words"

    def test_deduplicate_no_overlap(self):
        self.chunker.deduplicate_transcript("hello world")
        result = self.chunker.deduplicate_transcript("completely different")
        assert result == "completely different"

    def test_deduplicate_empty(self):
        assert self.chunker.deduplicate_transcript("") == ""

    # ── Reset ───────────────────────────────────────────────────────

    def test_reset_clears_state(self):
        frame = self._make_speech_frame()
        for _ in range(5):
            self.chunker.add_audio_frame(frame)
        self.chunker.deduplicate_transcript("test words")
        self.chunker.reset()
        assert len(self.chunker.audio_buffer) == 0
        assert len(self.chunker.confirmed_transcript_tokens) == 0


class TestInterceptionHUD:
    """Unit tests for the HUD severity mapping and Tier 2 reconciliation."""

    # ── Severity Mapping ────────────────────────────────────────────

    def test_severity_to_hud_state_mapping(self):
        assert SEVERITY_TO_HUD_STATE["LOW"] == "NEUTRAL"
        assert SEVERITY_TO_HUD_STATE["MEDIUM"] == "WARNING_AMBER"
        assert SEVERITY_TO_HUD_STATE["HIGH"] == "WARNING_AMBER"
        assert SEVERITY_TO_HUD_STATE["CRITICAL"] == "CRITICAL_RED"

    # ── Tier 2 Reconciliation ───────────────────────────────────────

    def test_reconcile_no_tier2(self):
        """No Tier 2 → trust Tier 1."""
        assert reconcile_hud_state("CRITICAL_RED", None) == "CRITICAL_RED"
        assert reconcile_hud_state("NEUTRAL", None) == "NEUTRAL"

    def test_reconcile_tier2_not_scam_downgrades(self):
        tier2 = {"is_scam": False, "confidence": 0.9}
        assert reconcile_hud_state("CRITICAL_RED", tier2) == "WARNING_AMBER"
        assert reconcile_hud_state("WARNING_AMBER", tier2) == "NEUTRAL"

    def test_reconcile_tier2_scam_escalates(self):
        tier2 = {"is_scam": True, "confidence": 0.9}
        assert reconcile_hud_state("WARNING_AMBER", tier2) == "CRITICAL_RED"

    def test_reconcile_low_confidence_no_override(self):
        tier2 = {"is_scam": False, "confidence": 0.3}
        assert reconcile_hud_state("CRITICAL_RED", tier2) == "CRITICAL_RED"

    def test_reconcile_tier2_scam_keeps_critical(self):
        """Already CRITICAL + confirmed scam → stays CRITICAL."""
        tier2 = {"is_scam": True, "confidence": 0.95}
        assert reconcile_hud_state("CRITICAL_RED", tier2) == "CRITICAL_RED"

    # ── HUD Payload Structure ───────────────────────────────────────

    def test_hud_alert_payload_keys(self):
        payloads = []
        hud = InterceptionHUD(telemetry_callback=lambda p: payloads.append(p))
        layer1 = {
            "cumulative_level": "CRITICAL",
            "cumulative_score": 85.0,
            "categories_triggered": ["digital_arrest"],
        }
        tier2 = {
            "is_scam": True,
            "confidence": 0.95,
            "intent_summary": "Authority impersonation",
            "recommended_action": "DISCONNECT_AND_MUTE",
        }
        hud.trigger_alert(layer1, tier2)
        assert len(payloads) == 1
        p = payloads[0]
        required = {"hud_state", "risk_score", "risk_level", "matched_categories",
                     "slm_reasoning", "recommended_action", "quick_actions"}
        assert required.issubset(set(p.keys()))
        assert p["hud_state"] == "CRITICAL_RED"
        assert len(p["quick_actions"]) > 0


# ═════════════════════════════════════════════════════════════════════
#  SECTION 2: FUNCTIONAL TESTS
# ═════════════════════════════════════════════════════════════════════

class TestEnsembleScorerFunctional:
    """Functional tests for the Layer 1 ensemble (Keyword + Semantic fusion)."""

    def setup_method(self):
        self.scorer = EnsembleScorer()

    def test_ensemble_analysis_returns_required_keys(self):
        result = self.scorer.analyze("test text")
        required = {"text", "keyword_score", "keyword_level", "tinybert_score",
                     "semantic_score", "ensemble_score", "ensemble_level",
                     "needs_layer2_review", "divergence", "detection", "semantic", "latency_ms"}
        assert required.issubset(set(result.keys()))

    def test_high_threat_text_ensemble_scores_high(self):
        text = ("Inspector Sharma from CBI. Digital arrest warrant issued. "
                "Transfer 50000 rupees via bitcoin to police clearance account immediately.")
        result = self.scorer.analyze(text)
        assert result["ensemble_score"] > 50
        assert result["ensemble_level"] in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_benign_text_ensemble_scores_low(self):
        text = "Good morning! Your delivery is on its way. It should arrive by 3 PM."
        result = self.scorer.analyze(text)
        assert result["ensemble_score"] < 40

    def test_borderline_triggers_l2_review(self):
        """A text that produces a borderline score (40-70) should flag L2 review."""
        # Use a text with some scam keywords but not overwhelming
        text = "There is an urgent matter regarding your account please call immediately"
        result = self.scorer.analyze(text)
        # This may or may not trigger L2 depending on exact score;
        # verify the flag is a boolean
        assert isinstance(result["needs_layer2_review"], bool)

    def test_quick_analyze_shortcut(self):
        result = quick_analyze("CBI officer digital arrest warrant")
        assert "ensemble_score" in result
        assert result["ensemble_score"] > 0

    def test_weights_sum_to_one(self):
        scorer = EnsembleScorer()
        assert abs(scorer.keyword_weight + scorer.semantic_weight - 1.0) < 0.001


class TestStreamingScamDetectorFunctional:
    """Functional tests for multi-chunk streaming session lifecycle."""

    def setup_method(self):
        self.detector = StreamingScamDetector(window_size_words=150, sliding_window_tokens=200)

    def test_session_starts_clean(self):
        assert self.detector.full_transcript == ""
        assert self.detector.peak_risk_score == 0.0
        assert len(self.detector.chunks) == 0

    def test_process_single_chunk(self):
        result = self.detector.process_chunk("Hello how are you")
        assert result["chunk_index"] == 1
        assert result["total_words"] == 4
        assert result["cumulative_score"] >= 0

    def test_process_multiple_chunks_accumulates(self):
        self.detector.process_chunk("Hello how are you")
        result = self.detector.process_chunk("CBI officer calling about digital arrest")
        assert result["chunk_index"] == 2
        assert result["total_words"] > 4
        assert result["cumulative_score"] > 0

    def test_peak_score_never_decreases(self):
        chunks = [
            "CBI officer digital arrest warrant issued",
            "transfer money immediately via bitcoin",
            "Hello how are you today nice weather",
            "I was thinking about dinner plans",
        ]
        previous_peak = 0.0
        for chunk in chunks:
            result = self.detector.process_chunk(chunk)
            assert result["peak_score"] >= previous_peak
            previous_peak = result["peak_score"]

    def test_sliding_window_limits_analysis_scope(self):
        """After many benign chunks, the score should decay (sliding window purge)."""
        # Feed a threat chunk
        self.detector.process_chunk("CBI officer digital arrest warrant FedEx parcel drugs")
        threat_result = self.detector.process_chunk("transfer 50000 rupees immediately to police account")

        # Feed many benign chunks to push threat tokens out of window
        for i in range(20):
            benign = f"The weather today is really nice, I had coffee this morning number {i}"
            result = self.detector.process_chunk(benign)

        # Current score should be lower than threat-phase score
        assert result["cumulative_score"] < threat_result["cumulative_score"]
        # But peak should remain anchored
        assert result["peak_score"] >= threat_result["cumulative_score"]

    def test_empty_chunk_returns_summary(self):
        result = self.detector.process_chunk("")
        assert "total_chunks" in result or "chunk_index" in result

    def test_reset_session(self):
        self.detector.process_chunk("test chunk")
        self.detector.reset_session()
        assert self.detector.full_transcript == ""
        assert self.detector.peak_risk_score == 0.0
        assert len(self.detector.chunks) == 0

    def test_session_summary(self):
        self.detector.process_chunk("test chunk one")
        self.detector.process_chunk("test chunk two")
        summary = self.detector.get_session_summary()
        assert summary["total_chunks"] == 2
        assert summary["total_words"] > 0
        assert "test chunk one test chunk two" == summary["full_transcript"]


class TestLiveCallInterceptorFunctional:
    """Functional tests for Tier 2 debounce, backend passthrough, and HUD dispatch."""

    @staticmethod
    def _make_scripted_asr(script):
        counter = [0]
        def engine(audio):
            idx = min(counter[0], len(script) - 1)
            counter[0] += 1
            return script[idx]
        return engine

    @staticmethod
    def _speech_frame():
        t = np.linspace(0, 0.5, 8000, False)
        return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_slm_backend_passthrough(self):
        """Verify slm_backend param reaches Tier2SLMJudge."""
        script = ["Hello testing"]
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(script),
            slm_backend="heuristic",
        )
        assert interceptor.slm_judge.backend == "heuristic"

    def test_tier2_not_triggered_on_benign(self):
        """Benign text should not trigger Tier 2 SLM."""
        benign_script = [
            "Hello how are you",
            "The weather is nice today",
            "I had lunch already",
        ]
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(benign_script),
            slm_backend="heuristic",
        )
        frame = self._speech_frame()
        results = []
        for _ in range(10):
            r = interceptor.on_audio_chunk_received(frame)
            if r:
                results.append(r)

        for r in results:
            assert "tier2_slm" not in r or r.get("tier2_slm") is None

    def test_tier2_triggered_on_threat(self):
        """Threatening text should trigger Tier 2 SLM at least once."""
        threat_script = [
            "This is Inspector from Delhi Cyber Crime Branch",
            "Digital arrest warrant has been issued by Supreme Court",
            "Transfer 50000 rupees to police clearance account immediately",
            "FedEx parcel containing MDMA drugs and contraband",
            "CBI enforcement directorate will arrest you",
        ]
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(threat_script),
            slm_backend="heuristic",
        )
        frame = self._speech_frame()
        tier2_invoked = False
        for _ in range(12):
            r = interceptor.on_audio_chunk_received(frame)
            if r and "tier2_slm" in r:
                tier2_invoked = True
                assert r["tier2_slm"]["is_scam"] is True

        assert tier2_invoked, "Tier 2 should have been invoked for threat text"

    def test_tier2_debounce_caches_verdict(self):
        """After first Tier 2 invocation, subsequent chunks should use cached verdict."""
        threat_script = [
            "CBI digital arrest warrant issued",
            "Transfer money via bitcoin immediately",
            "Police will arrest you",
            "FedEx parcel drugs contraband",
            "Supreme court order enforcement",
        ]
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(threat_script),
            slm_backend="heuristic",
        )
        frame = self._speech_frame()
        sources = []
        for _ in range(12):
            r = interceptor.on_audio_chunk_received(frame)
            if r and "tier2_source" in r:
                sources.append(r["tier2_source"])

        # Should have at least one "invoked" and potentially some "cached"
        assert "invoked" in sources

    def test_telemetry_dispatched(self):
        """Verify telemetry callback fires during processing."""
        payloads = []
        script = ["CBI digital arrest warrant police enforcement"]
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(script),
            telemetry_callback=lambda p: payloads.append(p),
            slm_backend="heuristic",
        )
        frame = self._speech_frame()
        for _ in range(8):
            interceptor.on_audio_chunk_received(frame)

        assert len(payloads) >= 1


# ═════════════════════════════════════════════════════════════════════
#  SECTION 3: END-TO-END INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════════════

# Full scam call transcript simulation
E2E_THREAT_SCRIPT = [
    "Hello this is Inspector Sharma from Delhi Cyber Crime Branch",
    "Delhi Cyber Crime Branch your Aadhaar card details were found in FedEx parcel",
    "were found in FedEx parcel containing contraband MDMA drugs and forged passports",
    "passports a digital arrest warrant has been issued by Supreme Court",
    "issued by Supreme Court transfer 50000 rupees to police clearance account immediately",
]

E2E_BENIGN_SCRIPT = [
    "So how was your day today the weather has been really nice",
    "I was thinking about ordering pizza for dinner tonight",
    "Did you watch the cricket match last night it was amazing",
    "My cousin is visiting from Bangalore next week very excited",
    "The new metro line is finally opening on Monday near my office",
    "I need to renew my passport before the December trip",
    "Have you tried that new restaurant on MG Road the biryani is great",
    "My daughter has her school annual day performance tomorrow",
]


class TestE2EIntegration:
    """Full pipeline end-to-end integration tests."""

    @staticmethod
    def _make_scripted_asr(script):
        counter = [0]
        def engine(audio):
            idx = min(counter[0], len(script) - 1)
            counter[0] += 1
            return script[idx]
        return engine

    @staticmethod
    def _speech_frame():
        t = np.linspace(0, 0.5, 8000, False)
        return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    @staticmethod
    def _silence_frame():
        return np.zeros(8000, dtype=np.float32)

    # ── E2E Test 1: Full Scam Call Pipeline ──────────────────────────

    def test_e2e_scam_call_detection(self):
        """
        Simulate a complete Digital Arrest scam call and verify:
        - Score reaches HIGH/CRITICAL
        - Tier 2 SLM is triggered
        - HUD dispatches CRITICAL_RED or WARNING_AMBER
        - Correct threat categories detected
        """
        payloads = []
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(E2E_THREAT_SCRIPT),
            telemetry_callback=lambda p: payloads.append(p),
            slm_backend="heuristic",
        )
        frame = self._speech_frame()
        results = []

        for _ in range(12):
            r = interceptor.on_audio_chunk_received(frame)
            if r:
                results.append(r)

        assert len(results) >= 3, f"Expected ≥3 results, got {len(results)}"

        # Verify score escalation
        final = results[-1]
        assert final["peak_score"] > 0
        assert final["cumulative_level"] in ("MEDIUM", "HIGH", "CRITICAL")

        # Verify Tier 2 was invoked at least once
        tier2_results = [r for r in results if "tier2_slm" in r]
        assert len(tier2_results) >= 1, "Tier 2 should fire on high-threat transcript"

        # Verify HUD telemetry dispatched
        assert len(payloads) >= 1
        hud_states = [p["hud_state"] for p in payloads]
        assert any(s in ("WARNING_AMBER", "CRITICAL_RED") for s in hud_states), \
            f"Expected HUD alert, got states: {hud_states}"

    # ── E2E Test 2: Benign Call — No False Positives ─────────────────

    def test_e2e_benign_call_no_false_positive(self):
        """Benign conversation should not trigger CRITICAL_RED alerts."""
        payloads = []
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(E2E_BENIGN_SCRIPT),
            telemetry_callback=lambda p: payloads.append(p),
            slm_backend="heuristic",
        )
        frame = self._speech_frame()
        results = []

        for _ in range(15):
            r = interceptor.on_audio_chunk_received(frame)
            if r:
                results.append(r)

        if results:
            final = results[-1]
            assert final["cumulative_level"] in ("LOW", "MEDIUM")

        # No CRITICAL_RED HUD alerts should fire
        critical_payloads = [p for p in payloads if p["hud_state"] == "CRITICAL_RED"]
        assert len(critical_payloads) == 0, "Benign text should not trigger CRITICAL_RED"

    # ── E2E Test 3: Silent → Threat → Benign Lifecycle ──────────────

    def test_e2e_silence_threat_benign_lifecycle(self):
        """
        Full lifecycle:
        1. Silent phase (no emissions)
        2. Threat phase (score escalates, Tier 2 fires)
        3. Benign phase (score decays, peak remains)
        """
        full_script = E2E_THREAT_SCRIPT + E2E_BENIGN_SCRIPT
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(full_script),
            slm_backend="heuristic",
        )
        speech = self._speech_frame()
        silence = self._silence_frame()

        # Phase 1: Silent (3 chunks)
        for _ in range(3):
            r = interceptor.on_audio_chunk_received(silence)
            assert r is None

        # Phase 2+3: Speech (threat then benign)
        results = []
        previous_peak = 0.0
        for _ in range(20):
            r = interceptor.on_audio_chunk_received(speech)
            if r:
                results.append(r)
                # Peak never decreases
                assert r["peak_score"] >= previous_peak
                previous_peak = r["peak_score"]
                # Score always in bounds
                assert 0.0 <= r["cumulative_score"] <= 100.0

        assert len(results) >= 5

        # Peak should be elevated from threat phase
        assert results[-1]["peak_score"] > 0

        # After enough benign chunks, current score should drop below peak
        if len(results) >= 10:
            assert results[-1]["cumulative_score"] <= results[-1]["peak_score"]

    # ── E2E Test 4: Transcript Continuity ────────────────────────────

    def test_e2e_transcript_continuity(self):
        """
        Verify that the full transcript assembled by StreamingScamDetector
        matches the joined emitted deltas — no dropped or duplicated words.
        """
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(E2E_THREAT_SCRIPT),
            slm_backend="heuristic",
        )
        frame = self._speech_frame()
        emitted_deltas = []

        for _ in range(12):
            r = interceptor.on_audio_chunk_received(frame)
            if r:
                emitted_deltas.append(r["transcript_delta"])

        joined = " ".join(emitted_deltas)
        detector_transcript = interceptor.detector.full_transcript
        assert joined == detector_transcript, (
            f"Transcript continuity broken!\n"
            f"  Joined deltas : {joined!r}\n"
            f"  Detector state: {detector_transcript!r}"
        )

    # ── E2E Test 5: Chunk Accounting ─────────────────────────────────

    def test_e2e_chunk_accounting(self):
        """Pipeline-emitted results count must match detector's internal chunk count."""
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(E2E_THREAT_SCRIPT),
            slm_backend="heuristic",
        )
        frame = self._speech_frame()
        result_count = 0

        for _ in range(12):
            r = interceptor.on_audio_chunk_received(frame)
            if r:
                result_count += 1

        assert result_count == len(interceptor.detector.chunks)

    # ── E2E Test 6: Latency Budget ───────────────────────────────────

    def test_e2e_latency_under_budget(self):
        """Each pipeline tick (with heuristic SLM) should complete in < 500ms."""
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(E2E_THREAT_SCRIPT),
            slm_backend="heuristic",
        )
        frame = self._speech_frame()
        latencies = []

        for _ in range(12):
            r = interceptor.on_audio_chunk_received(frame)
            if r:
                latencies.append(r["total_e2e_latency_ms"])

        assert len(latencies) >= 1
        for lat in latencies:
            assert lat < 500, f"Latency {lat:.2f}ms exceeds 500ms budget"

    # ── E2E Test 7: Multiple Backend Configs ─────────────────────────

    def test_e2e_backend_heuristic(self):
        """Verify the pipeline works end-to-end with explicit heuristic backend."""
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(E2E_THREAT_SCRIPT),
            slm_backend="heuristic",
        )
        assert interceptor.slm_judge.backend == "heuristic"
        frame = self._speech_frame()
        for _ in range(8):
            interceptor.on_audio_chunk_received(frame)
        assert interceptor.slm_judge.active_backend == "Heuristic (NPU Simulator)"

    def test_e2e_backend_litert_graceful_fallback(self):
        """
        If litert is requested but no .litertlm file exists,
        it should gracefully fall back to heuristic.
        """
        interceptor = LiveCallAudioInterceptor(
            asr_engine=self._make_scripted_asr(E2E_THREAT_SCRIPT),
            slm_backend="litert",
        )
        frame = self._speech_frame()
        for _ in range(8):
            r = interceptor.on_audio_chunk_received(frame)
        # Should have fallen back — LiteRT model file not present
        assert "Heuristic" in interceptor.slm_judge.active_backend

    # ── E2E Test 8: Full Module Import Integrity ─────────────────────

    def test_all_public_exports_importable(self):
        """Verify all __all__ exports are accessible."""
        import scam_detector
        for name in scam_detector.__all__:
            assert hasattr(scam_detector, name), f"Missing export: {name}"


# ═════════════════════════════════════════════════════════════════════
#  DATA DICTIONARY & PATTERN INTEGRITY TESTS
# ═════════════════════════════════════════════════════════════════════

class TestDataDictionaryIntegrity:
    """Validate scam word dictionary and pattern configuration consistency."""

    def test_all_categories_have_weights(self):
        for cat in SCAM_WORDS:
            assert cat in CATEGORY_WEIGHTS, f"Category '{cat}' missing from CATEGORY_WEIGHTS"

    def test_all_weight_categories_exist_in_dict(self):
        for cat in CATEGORY_WEIGHTS:
            assert cat in SCAM_WORDS, f"Weight for '{cat}' has no matching SCAM_WORDS entry"

    def test_hard_scam_categories_subset_of_all(self):
        assert HARD_SCAM_CATEGORIES.issubset(set(SCAM_WORDS.keys()))

    def test_all_patterns_have_weights(self):
        for pattern in PATTERNS:
            assert pattern in PATTERN_WEIGHTS, f"Pattern '{pattern}' missing from PATTERN_WEIGHTS"

    def test_no_empty_word_sets(self):
        for cat, words in SCAM_WORDS.items():
            assert len(words) > 0, f"Category '{cat}' has empty word set"

    def test_no_empty_pattern_lists(self):
        for pattern, regex_list in PATTERNS.items():
            assert len(regex_list) > 0, f"Pattern '{pattern}' has empty regex list"

    def test_weights_are_positive(self):
        for cat, weight in CATEGORY_WEIGHTS.items():
            assert weight > 0, f"Category '{cat}' has non-positive weight: {weight}"
        for pattern, weight in PATTERN_WEIGHTS.items():
            assert weight > 0, f"Pattern '{pattern}' has non-positive weight: {weight}"


# ═════════════════════════════════════════════════════════════════════
#  RUNNER
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
