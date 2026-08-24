"""
Scam Call Detection System
A local Python-based system to detect scam indicators in call transcripts.

Layer 1: Keywords + Semantic Ensemble
Layer 2: SLM Review
"""

from .detector import ScamDetector
from .scorer import RiskScorer, RiskLevel
from .word_dictionary import SCAM_WORDS, CATEGORY_WEIGHTS
from .tinybert_analyzer import TinyBertAnalyzer, SemanticAnalyzer
from .ensemble_scorer import EnsembleScorer, quick_analyze
from .streaming import StreamingScamDetector

__version__ = "2.0.0"
__all__ = [
    "ScamDetector", 
    "RiskScorer", 
    "RiskLevel",
    "SCAM_WORDS",
    "CATEGORY_WEIGHTS",
    "SemanticAnalyzer",
    "TinyBertAnalyzer",
    "EnsembleScorer",
    "StreamingScamDetector",
    "quick_analyze",
]
