"""
HUD & Action Controller
Dispatches visual/haptic warning states and telemetry feeds.
"""

from typing import Callable, Dict, Optional


class InterceptionHUD:
    def __init__(self, telemetry_callback: Optional[Callable[[Dict], None]] = None):
        self.telemetry_callback = telemetry_callback

    def trigger_alert(self, layer1_result: Dict, tier2_result: Optional[Dict] = None):
        """
        Emits real-time overlay actions to the mobile display and judging dashboard.
        """
        level = layer1_result.get("cumulative_level", "LOW")
        score = layer1_result.get("cumulative_score", 0.0)

        payload = {
            "hud_state": "NEUTRAL" if level == "LOW" else ("WARNING_AMBER" if level == "HIGH" else "CRITICAL_RED"),
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
