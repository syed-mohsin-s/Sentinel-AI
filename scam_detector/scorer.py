"""
Risk Scoring Algorithm
Calculates scam risk score based on detection results.
Includes false-positive caps for telemarketing/sales calls lacking hard scam vectors.
"""

from typing import Dict, Tuple
from .word_dictionary import CATEGORY_WEIGHTS, HARD_SCAM_CATEGORIES
from .patterns import PATTERN_WEIGHTS


class RiskLevel:
    """Risk level constants."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskScorer:
    """
    Calculate risk scores from scam detection analysis.
    """
    
    # Thresholds for risk levels (based on normalized score 0-100)
    THRESHOLDS = {
        RiskLevel.LOW: 0,
        RiskLevel.MEDIUM: 25,
        RiskLevel.HIGH: 50,
        RiskLevel.CRITICAL: 75,
    }
    
    # Bonus multipliers for category combinations
    COMBINATION_BONUSES = {
        ("digital_arrest", "urgency"): 1.6,
        ("digital_arrest", "financial"): 1.6,
        ("upi_coercion", "personal_info"): 1.7,
        ("fake_kyc", "personal_info"): 1.6,
        ("fake_kyc", "suspicious_actions"): 1.6,
        ("threats", "urgency"): 1.5,
        ("threats", "financial"): 1.4,
        ("impersonation", "personal_info"): 1.5,
        ("impersonation", "financial"): 1.4,
        ("urgency", "financial"): 1.3,
        ("prize_lottery", "personal_info"): 1.4,
    }
    
    def __init__(self):
        self.category_weights = CATEGORY_WEIGHTS
        self.pattern_weights = PATTERN_WEIGHTS
    
    def calculate_word_score(self, detected_words: Dict) -> float:
        """
        Calculate score from detected suspicious words.
        """
        score = 0.0
        
        for category, words in detected_words.items():
            weight = self.category_weights.get(category, 5)
            unique_words = set(word for word, _ in words)
            
            for i, _ in enumerate(unique_words):
                multiplier = 1.0 / (1 + i * 0.2)
                score += weight * multiplier
        
        return score
    
    def calculate_pattern_score(self, detected_patterns: Dict) -> float:
        """
        Calculate score from detected patterns.
        """
        score = 0.0
        
        for pattern_name, matches in detected_patterns.items():
            weight = self.pattern_weights.get(pattern_name, 5)
            for i in range(len(matches)):
                multiplier = 1.0 / (1 + i * 0.1)
                score += weight * multiplier
        
        return score
    
    def calculate_density_bonus(self, suspicious_count: int, total_words: int) -> float:
        """
        Calculate bonus based on density of suspicious words.
        """
        if total_words == 0:
            return 1.0
        
        density = suspicious_count / total_words
        return min(1.0 + (density * 10), 2.0)
    
    def calculate_combination_bonus(self, categories: list) -> float:
        """
        Calculate bonus for dangerous category combinations.
        """
        max_bonus = 1.0
        
        for combo, bonus in self.COMBINATION_BONUSES.items():
            if all(cat in categories for cat in combo):
                max_bonus = max(max_bonus, bonus)
        
        return max_bonus
    
    def calculate_score(self, analysis: Dict) -> Tuple[float, float, str]:
        """
        Calculate final risk score from analysis.
        
        Returns:
            Tuple of (raw_score, normalized_score, risk_level)
        """
        word_score = self.calculate_word_score(analysis['detected_words'])
        pattern_score = self.calculate_pattern_score(analysis['detected_patterns'])
        
        raw_score = word_score + pattern_score
        
        density_bonus = self.calculate_density_bonus(
            analysis['suspicious_word_count'],
            analysis['word_count']
        )
        
        combination_bonus = self.calculate_combination_bonus(
            analysis['categories_triggered']
        )
        
        final_raw = raw_score * density_bonus * combination_bonus
        
        # Normalize score (0-100)
        normalized = min((final_raw / 120.0) * 100, 100)
        
        # FALSE POSITIVE MITIGATION RULE:
        # Check if any hard scam vector is present (Digital Arrest, UPI coercion, Fake KYC, Threats, OTP/Personal Info, Financial, Suspicious Actions)
        triggered_cats = set(analysis['categories_triggered'])
        has_hard_scam_category = bool(triggered_cats.intersection(HARD_SCAM_CATEGORIES))
        has_pattern_match = len(analysis['detected_patterns']) > 0
        
        if not (has_hard_scam_category or has_pattern_match):
            # Text only has general pressure tactics or mild sales talk -> Cap risk at MEDIUM (max 35/100)
            normalized = min(normalized, 35.0)
        
        # Determine risk level
        risk_level = RiskLevel.LOW
        for level, threshold in sorted(self.THRESHOLDS.items(), 
                                        key=lambda x: x[1], reverse=True):
            if normalized >= threshold:
                risk_level = level
                break
        
        return final_raw, normalized, risk_level
    
    def get_risk_emoji(self, risk_level: str) -> str:
        """Get emoji for risk level."""
        return {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.HIGH: "🟠",
            RiskLevel.CRITICAL: "🔴",
        }.get(risk_level, "⚪")
    
    def get_score_report(self, analysis: Dict) -> str:
        """Generate a formatted risk score report."""
        raw_score, normalized, risk_level = self.calculate_score(analysis)
        emoji = self.get_risk_emoji(risk_level)
        
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("RISK ASSESSMENT")
        lines.append("=" * 60)
        
        bar_length = 40
        filled = int((normalized / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        lines.append(f"\n{emoji} Risk Level: {risk_level}")
        lines.append(f"\n   Score: [{bar}] {normalized:.1f}/100")
        lines.append(f"\n   Raw Score: {raw_score:.2f}")
        
        if analysis['categories_triggered']:
            lines.append(f"\n   Categories Detected: {len(analysis['categories_triggered'])}")
            lines.append(f"   → {', '.join(analysis['categories_triggered'])}")
        
        lines.append("\n" + "-" * 60)
        if risk_level == RiskLevel.CRITICAL:
            lines.append("⚠️  CRITICAL: High-risk scam indicators detected!")
            lines.append("   Immediate threat vector detected (Digital Arrest / UPI / OTP / Lawsuit).")
        elif risk_level == RiskLevel.HIGH:
            lines.append("⚠️  HIGH RISK: Significant scam indicators present.")
            lines.append("   Exercise extreme caution.")
        elif risk_level == RiskLevel.MEDIUM:
            lines.append("⚠️  MEDIUM RISK: Some suspicious elements or marketing pressure.")
            lines.append("   Review carefully before proceeding.")
        else:
            lines.append("✅ LOW RISK: Few or no scam indicators detected.")
            lines.append("   Text appears safe.")
        
        lines.append("=" * 60)
        return "\n".join(lines)
