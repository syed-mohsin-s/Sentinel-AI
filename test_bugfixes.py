"""
Acceptance Tests for Bug Fixes 1-4.

Run with: python -m pytest test_bugfixes.py -v
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pytest
from scam_detector.interception_hud import (
    SEVERITY_TO_HUD_STATE,
    InterceptionHUD,
    reconcile_hud_state,
)
from scam_detector.tier2_slm_judge import Tier2SLMJudge
from scam_detector.tinybert_analyzer import (
    SemanticAnalyzer,
    TfIdfSemanticAnalyzer,
    TinyBertAnalyzer,
)


# ═══════════════════════════════════════════════════════════════════════
#  BUG 1 — Severity mapping must not be inverted
# ═══════════════════════════════════════════════════════════════════════


class TestBug1SeverityMapping:
    """Verify the explicit SEVERITY_TO_HUD_STATE mapping covers all levels."""

    def test_low_maps_to_neutral(self):
        assert SEVERITY_TO_HUD_STATE["LOW"] == "NEUTRAL"

    def test_medium_maps_to_warning_amber(self):
        """This was the core bug — MEDIUM previously fell through to CRITICAL_RED."""
        assert SEVERITY_TO_HUD_STATE["MEDIUM"] == "WARNING_AMBER"

    def test_high_maps_to_warning_amber(self):
        assert SEVERITY_TO_HUD_STATE["HIGH"] == "WARNING_AMBER"

    def test_critical_maps_to_critical_red(self):
        assert SEVERITY_TO_HUD_STATE["CRITICAL"] == "CRITICAL_RED"

    def test_unknown_level_defaults_to_neutral(self):
        """Unknown level must fail safe (NEUTRAL), never CRITICAL_RED."""
        assert SEVERITY_TO_HUD_STATE.get("BANANA", "NEUTRAL") == "NEUTRAL"

    def test_hud_trigger_with_medium_level(self):
        """End-to-end: a MEDIUM-level Layer 1 result must produce WARNING_AMBER, not CRITICAL_RED."""
        hud = InterceptionHUD()
        payloads = []
        hud.telemetry_callback = lambda p: payloads.append(p)

        hud.trigger_alert({"cumulative_level": "MEDIUM", "cumulative_score": 34.3})
        assert payloads[0]["hud_state"] == "WARNING_AMBER"

    def test_hud_trigger_with_unknown_level(self):
        """Unknown level must not crash and must default to NEUTRAL."""
        hud = InterceptionHUD()
        payloads = []
        hud.telemetry_callback = lambda p: payloads.append(p)

        hud.trigger_alert({"cumulative_level": "XYZZY", "cumulative_score": 50.0})
        assert payloads[0]["hud_state"] == "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════
#  BUG 2 — executed_on_npu must reflect what actually happened
# ═══════════════════════════════════════════════════════════════════════


class TestBug2NPUFlag:
    """executed_on_npu must be False when the heuristic fallback ran,
    even if use_npu=True was passed to the constructor."""

    def test_heuristic_backend_reports_npu_false(self):
        judge = Tier2SLMJudge(use_npu=True)
        # Force heuristic backend (no torch/ollama/llama_cpp/onnx available)
        judge.active_backend = "Heuristic (NPU Simulator)"
        judge.is_loaded = True

        result = judge.evaluate_transcript("Hello, this is a normal call.")
        assert result["executed_on_npu"] is False, (
            f"Heuristic backend must report executed_on_npu=False, got {result['executed_on_npu']}"
        )

    def test_backend_field_is_present(self):
        judge = Tier2SLMJudge(use_npu=True)
        result = judge.evaluate_transcript("Hello, this is a normal call.")
        assert "backend" in result, "Result must include 'backend' key"
        assert isinstance(result["backend"], str)

    def test_use_npu_true_does_not_leak_into_output(self):
        """Even with use_npu=True, if no real NPU backend loaded,
        executed_on_npu must be False."""
        judge = Tier2SLMJudge(use_npu=True, backend="heuristic")
        result = judge.evaluate_transcript("transfer money now or face arrest warrant")
        assert result["executed_on_npu"] is False


# ═══════════════════════════════════════════════════════════════════════
#  BUG 3 — No "TinyBERT" / "BERT" in output when TF-IDF is the backend
# ═══════════════════════════════════════════════════════════════════════


class TestBug3TfIdfLabel:
    """The backend label must honestly say TF-IDF, not TinyBERT/BERT."""

    def test_backend_label_is_tfidf(self):
        analyzer = TfIdfSemanticAnalyzer()
        result = analyzer.analyze("Hello, is this a scam?")
        backend = result["backend"]
        assert "BERT" not in backend.upper(), (
            f"Backend label must not mention BERT, got '{backend}'"
        )
        assert "TF-IDF" in backend, (
            f"Backend label must say 'TF-IDF', got '{backend}'"
        )

    def test_semantic_analyzer_alias_works(self):
        """SemanticAnalyzer is a backwards-compatible alias."""
        analyzer = SemanticAnalyzer()
        result = analyzer.analyze("your account has been compromised")
        assert "TF-IDF" in result["backend"]

    def test_tinybert_alias_works(self):
        """TinyBertAnalyzer alias still works for backwards compatibility."""
        analyzer = TinyBertAnalyzer()
        result = analyzer.analyze("you have won a lottery")
        assert "TF-IDF" in result["backend"]
        assert "BERT" not in result["backend"].upper()

    def test_no_model_name_attribute(self):
        """model_name should no longer exist since no model is loaded."""
        analyzer = TfIdfSemanticAnalyzer()
        assert not hasattr(analyzer, "model_name"), (
            "model_name attribute should be removed — no transformer model is loaded"
        )


# ═══════════════════════════════════════════════════════════════════════
#  BUG 4 — Tier 2's verdict must be able to override Tier 1's HUD
# ═══════════════════════════════════════════════════════════════════════


class TestBug4Reconciliation:
    """Tier 2 SLM verdict must be able to downgrade or escalate the HUD state."""

    def test_tier2_none_preserves_tier1(self):
        assert reconcile_hud_state("CRITICAL_RED", None) == "CRITICAL_RED"
        assert reconcile_hud_state("WARNING_AMBER", None) == "WARNING_AMBER"
        assert reconcile_hud_state("NEUTRAL", None) == "NEUTRAL"

    def test_tier2_confident_not_scam_downgrades_critical(self):
        """Tier 2 says 'not a scam' with 90% confidence → downgrade CRITICAL → AMBER."""
        result = reconcile_hud_state(
            "CRITICAL_RED", {"is_scam": False, "confidence": 0.9}
        )
        assert result == "WARNING_AMBER", f"Expected WARNING_AMBER, got {result}"

    def test_tier2_confident_not_scam_downgrades_amber(self):
        """Tier 2 says 'not a scam' with 80% confidence → downgrade AMBER → NEUTRAL."""
        result = reconcile_hud_state(
            "WARNING_AMBER", {"is_scam": False, "confidence": 0.80}
        )
        assert result == "NEUTRAL", f"Expected NEUTRAL, got {result}"

    def test_tier2_confident_scam_escalates_amber(self):
        """Tier 2 confirms scam with 95% confidence → escalate AMBER → CRITICAL."""
        result = reconcile_hud_state(
            "WARNING_AMBER", {"is_scam": True, "confidence": 0.95}
        )
        assert result == "CRITICAL_RED", f"Expected CRITICAL_RED, got {result}"

    def test_tier2_low_confidence_does_not_override(self):
        """Low-confidence Tier 2 verdict must not change anything."""
        result = reconcile_hud_state(
            "CRITICAL_RED", {"is_scam": False, "confidence": 0.5}
        )
        assert result == "CRITICAL_RED", f"Low-confidence should not override, got {result}"

    def test_tier2_confident_not_scam_does_not_downgrade_neutral(self):
        """NEUTRAL can't be downgraded further."""
        result = reconcile_hud_state(
            "NEUTRAL", {"is_scam": False, "confidence": 0.9}
        )
        assert result == "NEUTRAL"

    def test_tier2_confident_scam_keeps_critical(self):
        """Already CRITICAL + scam confirmed → stays CRITICAL (no double-escalation)."""
        result = reconcile_hud_state(
            "CRITICAL_RED", {"is_scam": True, "confidence": 0.95}
        )
        assert result == "CRITICAL_RED"

    def test_end_to_end_hud_with_tier2_downgrade(self):
        """Full HUD trigger_alert: Tier 1 CRITICAL + Tier 2 'not scam' → WARNING_AMBER."""
        hud = InterceptionHUD()
        payloads = []
        hud.telemetry_callback = lambda p: payloads.append(p)

        tier2 = {"is_scam": False, "confidence": 0.9,
                 "intent_summary": "Benign call", "recommended_action": "ALLOW_CALL"}
        hud.trigger_alert(
            {"cumulative_level": "CRITICAL", "cumulative_score": 90.0},
            tier2_result=tier2,
        )
        assert payloads[0]["hud_state"] == "WARNING_AMBER"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
