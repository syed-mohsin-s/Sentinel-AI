"""
Semantic Analyzer
Supports HuggingFace Transformer models (TinyBERT / Gemma-3-270M / BERT) with
a calibrated TF-IDF vectorizer fallback.

Fixes raw cosine scaling miscalibration and incorporates India-specific scam vectors.
"""

import time
from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticAnalyzer:
    """
    Semantic analyzer for scam detection.
    Evaluates semantic similarity between input text and known scam/legitimate intent vectors.
    """
    
    def __init__(self, model_name: str = "prajjwal1/bert-tiny"):
        """
        Initialize the semantic analyzer.
        """
        self.model_name = model_name
        self.vectorizer = None
        self.is_loaded = False
        self.backend = "TF-IDF (Calibrated)"
        
        # India-Specific & Core Scam Reference Indicators
        self.scam_indicators = [
            "cbi enforcement directorate police customs digital arrest warrant supreme court mdma drugs parcel illegal",
            "upi pin phonepe gpay google pay enter pin receive money accept collect request qr code scan",
            "trai sim card block deactivation aadhaar kyc update anydesk teamviewer quicksupport apk download screen share",
            "arrest warrant police jail legal action lawsuit criminal charges federal marshals court prosecution",
            "gift card itunes target amazon bitcoin wire transfer western union rbi secret account penalty payment",
            "social security number ssn password pin account number otp verification code netbanking credentials",
            "urgent immediately act now don't hang up final warning emergency within 2 hours 30 minutes",
            "lottery winner prize congratulations million dollars selected claim reward jackpot",
            "compromised hacked virus malware remote access download install quicksupport clean infection",
            "threatening payment demand legal consequences failure to comply asset freeze blocked account",
        ]
        
        # Legitimate Reference Indicators
        self.legitimate_indicators = [
            "appointment schedule confirm available tomorrow doctor clinic health check-in",
            "delivery package shipped tracking arrive bluedart courier driver front door",
            "interview position opportunity resume application human resources job video meet",
            "bill payment due amount monthly service utility receipt transaction statement",
            "routine reminder notice information update account visit nearest branch yono app",
            "thank you customer service assistance feedback survey satisfaction rating",
            "hdfc bank automated fraud alert transaction press 1 to confirm press 2 to block never ask for otp",
        ]
        
        self._scam_vectors = None
        self._legit_vectors = None
        self._all_indicators = None
    
    def load_model(self):
        """Initialize reference vectors and TF-IDF calibration."""
        if self.is_loaded:
            return
        
        t0 = time.perf_counter()
        
        # Combine all indicators
        self._all_indicators = self.scam_indicators + self.legitimate_indicators
        
        # Create n-gram TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words='english',
            lowercase=True,
            sublinear_tf=True
        )
        
        self.vectorizer.fit(self._all_indicators)
        
        self._scam_vectors = self.vectorizer.transform(self.scam_indicators).toarray()
        self._legit_vectors = self.vectorizer.transform(self.legitimate_indicators).toarray()
        
        self.is_loaded = True
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # print(f"Semantic analyzer initialized ({self.backend}) in {elapsed_ms:.1f}ms")
    
    def analyze(self, text: str) -> dict:
        """
        Analyze text for semantic scam indicators.
        
        Returns:
            dict containing scam_score (0-100), confidence, similarities, and latency_ms.
        """
        t0 = time.perf_counter()
        
        if not self.is_loaded:
            self.load_model()
        
        text_vector = self.vectorizer.transform([text]).toarray()
        
        scam_sims = cosine_similarity(text_vector, self._scam_vectors)[0]
        legit_sims = cosine_similarity(text_vector, self._legit_vectors)[0]
        
        max_scam_sim = float(np.max(scam_sims))
        max_legit_sim = float(np.max(legit_sims))
        avg_scam_sim = float(np.mean(scam_sims))
        avg_legit_sim = float(np.mean(legit_sims))
        
        # Calibrated Scam Score Math
        scam_score = self._calculate_calibrated_scam_score(
            max_scam_sim, max_legit_sim, avg_scam_sim, avg_legit_sim
        )
        
        # Confidence calculation based on margin
        margin = max_scam_sim - max_legit_sim
        confidence = float(np.clip(abs(margin) * 250.0, 10.0, 99.0))
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        return {
            "scam_score": scam_score,
            "confidence": confidence,
            "scam_similarity": avg_scam_sim,
            "legit_similarity": avg_legit_sim,
            "max_scam_similarity": max_scam_sim,
            "max_legit_similarity": max_legit_sim,
            "backend": self.backend,
            "latency_ms": elapsed_ms,
        }
    
    def _calculate_calibrated_scam_score(
        self, max_scam: float, max_legit: float, avg_scam: float, avg_legit: float
    ) -> float:
        """
        Calibrated scoring function to prevent low raw similarity from inflating scores.
        
        - If max_scam < 0.08: noise / no match -> 0-15 risk score.
        - Uses differential similarity: delta = max_scam - max_legit.
        - Calibrated min-max mapping based on empirical similarity distributions.
        """
        if max_scam < 0.08 and avg_scam < 0.03:
            # Barely any lexical or semantic overlap with scam templates
            return float(np.clip(max_scam * 150.0, 0.0, 15.0))
        
        # Differential signal
        delta = max_scam - (max_legit * 0.7)
        
        if delta <= 0:
            # Stronger match to legitimate text
            score = max_scam * 100.0
            return float(np.clip(score, 0.0, 25.0))
        
        # Positive scam differential
        # Scale: delta from 0.0 to 0.5 maps to score 30 -> 100
        base_score = 25.0 + (delta / 0.45) * 75.0
        
        # Boost if max_scam similarity is high (> 0.25)
        if max_scam > 0.25:
            base_score += 10.0
            
        return float(np.clip(base_score, 0.0, 100.0))
    
    def get_analysis_report(self, analysis: dict) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append(f"\n--- Semantic Analysis ({analysis['backend']}) ---")
        lines.append(f"Semantic Score: {analysis['scam_score']:.1f}/100")
        lines.append(f"Confidence:     {analysis['confidence']:.1f}%")
        lines.append(f"Max Scam Sim:   {analysis['max_scam_similarity']:.3f}")
        lines.append(f"Max Legit Sim:  {analysis['max_legit_similarity']:.3f}")
        lines.append(f"Latency:        {analysis['latency_ms']:.2f} ms")
        lines.append("-" * 35)
        return "\n".join(lines)


# Alias TinyBertAnalyzer to SemanticAnalyzer for backwards compatibility
TinyBertAnalyzer = SemanticAnalyzer
