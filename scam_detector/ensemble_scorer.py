"""
Ensemble Scorer (Layer 1)
Combines Keyword/Pattern Scorer (40%) and Semantic Analyzer (60%).
Flags calls needing Layer 2 SLM review and measures end-to-end latency.
"""

import time
from typing import Dict
from .detector import ScamDetector
from .scorer import RiskScorer, RiskLevel
from .tinybert_analyzer import TinyBertAnalyzer


class EnsembleScorer:
    """
    Layer 1 Ensemble Scorer combining Keyword Rules + Semantic Embeddings.
    """
    
    def __init__(self, keyword_weight: float = 0.4, semantic_weight: float = 0.6):
        """
        Initialize ensemble scorer.
        
        Args:
            keyword_weight: Weight for keyword/pattern risk score (default 0.4)
            semantic_weight: Weight for semantic analyzer risk score (default 0.6)
        """
        self.detector = ScamDetector()
        self.scorer = RiskScorer()
        self.semantic_analyzer = TinyBertAnalyzer()
        
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
    
    def analyze(self, text: str) -> Dict:
        """
        Perform complete Layer 1 Ensemble Analysis.
        
        Returns:
            Dict containing keyword score, semantic score, ensemble score, risk level,
            layer2 flag, details, and end-to-end latency_ms.
        """
        t0 = time.perf_counter()
        
        # 1. Run Keyword & Pattern Detection
        detection_result = self.detector.analyze(text)
        raw_kw_score, keyword_score, kw_level = self.scorer.calculate_score(detection_result)
        
        # 2. Run Semantic Analysis
        semantic_result = self.semantic_analyzer.analyze(text)
        semantic_score = semantic_result['scam_score']
        
        # 3. Compute Ensemble Weighted Score
        ensemble_score = (keyword_score * self.keyword_weight) + (semantic_score * self.semantic_weight)
        
        # 4. Determine Ensemble Risk Level
        ensemble_level = RiskLevel.LOW
        for level, threshold in sorted(self.scorer.THRESHOLDS.items(), 
                                        key=lambda x: x[1], reverse=True):
            if ensemble_score >= threshold:
                ensemble_level = level
                break
        
        # 5. Layer 2 SLM Review Trigger Rule
        # Trigger L2 if score is in borderline range (40 - 70) or if Kw and Semantic diverge by > 40 points
        divergence = abs(keyword_score - semantic_score)
        needs_l2 = (40.0 <= ensemble_score <= 70.0) or (divergence >= 40.0 and ensemble_score >= 30.0)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        return {
            "text": text,
            "keyword_score": keyword_score,
            "keyword_level": kw_level,
            "tinybert_score": semantic_score, # Retained key for test compatibility
            "semantic_score": semantic_score,
            "ensemble_score": ensemble_score,
            "ensemble_level": ensemble_level,
            "needs_layer2_review": needs_l2,
            "divergence": divergence,
            "detection": detection_result,
            "semantic": semantic_result,
            "latency_ms": elapsed_ms,
        }
    
    def get_report(self, analysis: Dict) -> str:
        """Generate formatted ensemble analysis report."""
        lines = []
        lines.append("\n" + "=" * 65)
        lines.append("  LAYER 1 ENSEMBLE ASSESSMENT (40% KW + 60% Semantic)")
        lines.append("=" * 65)
        
        emoji = self.scorer.get_risk_emoji(analysis['ensemble_level'])
        
        bar_len = 30
        filled = int((analysis['ensemble_score'] / 100) * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        lines.append(f"\n{emoji} Ensemble Risk Level: {analysis['ensemble_level']}")
        lines.append(f"   Ensemble Score: [{bar}] {analysis['ensemble_score']:.1f}/100")
        lines.append(f"   Breakdown:      Keyword: {analysis['keyword_score']:.1f} | Semantic: {analysis['semantic_score']:.1f}")
        lines.append(f"   Latency:        {analysis['latency_ms']:.2f} ms")
        
        l2_status = "⚠️  YES (Proceed to Layer 2 SLM)" if analysis['needs_layer2_review'] else "✅ NO (Layer 1 Decisive)"
        lines.append(f"   Needs L2 Review: {l2_status}")
        
        det = analysis['detection']
        if det['categories_triggered']:
            lines.append(f"\n   Categories Triggered: {', '.join(det['categories_triggered'])}")
        if det['detected_patterns']:
            lines.append(f"   Patterns Matched:     {', '.join(det['detected_patterns'].keys())}")
            
        lines.append("=" * 65)
        return "\n".join(lines)


def quick_analyze(text: str) -> Dict:
    """Helper function for fast one-shot analysis."""
    scorer = EnsembleScorer()
    return scorer.analyze(text)
