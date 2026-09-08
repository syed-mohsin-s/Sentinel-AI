"""
HUD & Action Controller
Dispatches visual/haptic warning states and telemetry feeds.
"""

import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Explicit severity-to-HUD mapping — covers all four levels with no
# fallthrough "else" that can silently absorb an unhandled case.
SEVERITY_TO_HUD_STATE = {
    "LOW": "NEUTRAL",
    "MEDIUM": "WARNING_AMBER",
    "HIGH": "WARNING_AMBER",
    "CRITICAL": "CRITICAL_RED",
}


def reconcile_hud_state(tier1_hud_state: str, tier2_result: Optional[Dict]) -> str:
    """Reconcile Tier 1 HUD state against Tier 2 SLM verdict when available.

    Rules:
    - Tier 2 not available → trust Tier 1 alone.
    - Tier 2 confidently says NOT a scam (confidence >= 0.75) → downgrade one step
      (CRITICAL_RED→WARNING_AMBER, WARNING_AMBER→NEUTRAL), never fully silence.
    - Tier 2 confidently confirms scam (confidence >= 0.75) → escalate one step
      (WARNING_AMBER→CRITICAL_RED).
    - Low-confidence Tier 2 → don't override Tier 1.
    """
    if tier2_result is None:
        return tier1_hud_state  # Tier 2 hasn't run yet, trust Tier 1 alone

    is_scam = tier2_result.get("is_scam")
    confidence = tier2_result.get("confidence", 0.0)

    # Tier 2 confidently says NOT a scam → downgrade, don't fully
    # silence (still show amber, never silently drop a Tier-1-flagged
    # call straight to NEUTRAL — leave a visible but calmer signal)
    if is_scam is False and confidence >= 0.75:
        if tier1_hud_state == "CRITICAL_RED":
            logger.info("Tier 2 downgrade: CRITICAL_RED → WARNING_AMBER (is_scam=False, confidence=%.2f)", confidence)
            return "WARNING_AMBER"
        if tier1_hud_state == "WARNING_AMBER":
            logger.info("Tier 2 downgrade: WARNING_AMBER → NEUTRAL (is_scam=False, confidence=%.2f)", confidence)
            return "NEUTRAL"
        return tier1_hud_state

    # Tier 2 confidently confirms scam → keep or escalate
    if is_scam is True and confidence >= 0.75:
        if tier1_hud_state == "WARNING_AMBER":
            logger.info("Tier 2 escalation: WARNING_AMBER → CRITICAL_RED (is_scam=True, confidence=%.2f)", confidence)
            return "CRITICAL_RED"
        return tier1_hud_state

    # Low-confidence Tier 2 verdict → don't let it override anything
    return tier1_hud_state


class InterceptionHUD:
    def __init__(self, telemetry_callback: Optional[Callable[[Dict], None]] = None):
        self.telemetry_callback = telemetry_callback

    def trigger_alert(self, layer1_result: Dict, tier2_result: Optional[Dict] = None):
        """
        Emits real-time overlay actions to the mobile display and judging dashboard.
        """
        level = layer1_result.get("cumulative_level", "LOW")
        score = layer1_result.get("cumulative_score", 0.0)

        # Bug 1 fix: use explicit mapping instead of inverted ternary
        hud_state = SEVERITY_TO_HUD_STATE.get(level, "NEUTRAL")
        if level not in SEVERITY_TO_HUD_STATE:
            logger.warning("Unknown severity level %r — defaulting to NEUTRAL (fail safe)", level)

        # Bug 4 fix: reconcile Tier 1 HUD state against Tier 2 verdict
        hud_state = reconcile_hud_state(hud_state, tier2_result)

        payload = {
            "hud_state": hud_state,
            "risk_score": score,
            "risk_level": level,
            "matched_categories": layer1_result.get("categories_triggered", []),
            "slm_reasoning": tier2_result.get("intent_summary") if tier2_result else "Evaluating context...",
            "recommended_action": tier2_result.get("recommended_action") if tier2_result else "MONITOR",
            "quick_actions": ["MUTE_MIC", "SAFE_DISCONNECT", "LOG_CYBERCRIME_REPORT"] if level == "CRITICAL" or level == "HIGH" else []
        }

        self._render_terminal_hud(payload)

        if self.telemetry_callback:
            self.telemetry_callback(payload)

    def _render_terminal_hud(self, payload: Dict):
        if payload["hud_state"] == "CRITICAL_RED":
            print("\n" + "🚨" * 32)
            print("  [HUD OVERLAY: CRITICAL SCAM INTERCEPTION]")
            print(f"  Threat:    {payload['slm_reasoning']}")
            print(f"  Action:    {payload['recommended_action']} | Quick Controls: {', '.join(payload['quick_actions'])}")
            print("🚨" * 32 + "\n")
        elif payload["hud_state"] == "WARNING_AMBER":
            print("\n" + "⚠️ " * 32)
            print("  [HUD OVERLAY: AMBER WARNING - SLM REVIEW ACTIVE]")
            print(f"  Intent:    {payload['slm_reasoning']}")
            print(f"  Action:    {payload['recommended_action']}")
            print("⚠️ " * 32 + "\n")
