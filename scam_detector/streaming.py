"""
Streaming / Partial Transcript Analyzer
Handles live call audio transcripts arriving in real-time incremental chunks.
Maintains session state, rolling windows, and cumulative evidence accumulation.

Performance Note:
    Previous implementation called ensemble.analyze() on the FULL cumulative
    transcript every chunk, yielding O(N²) total TF-IDF vectorization cost.
    Now uses a fixed-size sliding window of the last K tokens so each chunk
    is O(K) regardless of total transcript length.
"""

from typing import Dict, List, Optional
from .ensemble_scorer import EnsembleScorer


class StreamingScamDetector:
    """
    Session-level streaming transcript analyzer for real-time call monitoring.
    Uses a sliding window of the last K tokens for constant-time-per-chunk
    vectorization instead of re-processing the entire cumulative transcript.
    """
    
    def __init__(
        self,
        window_size_words: int = 150,
        sliding_window_tokens: int = 200,
    ):
        """
        Initialize streaming session detector.
        
        Args:
            window_size_words: Legacy rolling window size (kept for API compat).
            sliding_window_tokens: Number of trailing tokens fed to the ensemble
                scorer each chunk.  ~200 tokens ≈ 100s of speech at 2 words/sec.
        """
        self.ensemble = EnsembleScorer()
        self.window_size_words = window_size_words
        self.sliding_window_tokens = sliding_window_tokens
        self.reset_session()
    
    def reset_session(self):
        """Reset session state for a new call."""
        self.chunks: List[str] = []
        self.full_transcript: str = ""
        self.word_tokens: List[str] = []
        self.peak_risk_score: float = 0.0
        self.peak_risk_level: str = "LOW"
        self.peak_categories: List[str] = []
        self.session_history: List[Dict] = []
    
    def process_chunk(self, chunk_text: str) -> Dict:
        """
        Process an incoming transcript chunk from live STT stream.
        
        Args:
            chunk_text: Text snippet of new audio transcription.
            
        Returns:
            Dict containing current_chunk_score, cumulative_score, peak_score,
            risk_level, and flags.
        """
        chunk_clean = chunk_text.strip()
        if not chunk_clean:
            return self.get_session_summary()
        
        self.chunks.append(chunk_clean)
        self.full_transcript = " ".join(self.chunks)
        self.word_tokens = self.full_transcript.split()
        
        # ── Sliding-window analysis (O(K) per chunk, not O(N)) ──────────
        # Use only the last K tokens for the ensemble scorer.  This keeps
        # vectorization cost constant as the transcript grows.
        window_tokens = self.word_tokens[-self.sliding_window_tokens:]
        window_text = " ".join(window_tokens)
        
        analysis = self.ensemble.analyze(window_text)
        
        current_score = analysis['ensemble_score']
        current_level = analysis['ensemble_level']
        
        # Track peak risk across the entire session so evidence from
        # earlier windows that have scrolled out is never lost.
        if current_score > self.peak_risk_score:
            self.peak_risk_score = current_score
            self.peak_risk_level = current_level
            self.peak_categories = analysis['detection']['categories_triggered']
        
        chunk_result = {
            "chunk_index": len(self.chunks),
            "chunk_text": chunk_clean,
            "total_words": len(self.word_tokens),
            "cumulative_score": current_score,
            "cumulative_level": current_level,
            "peak_score": self.peak_risk_score,
            "peak_level": self.peak_risk_level,
            "needs_l2_review": analysis['needs_layer2_review'],
            "categories_triggered": analysis['detection']['categories_triggered'],
            "latency_ms": analysis['latency_ms'],
        }
        
        self.session_history.append(chunk_result)
        return chunk_result
    
    def get_session_summary(self) -> Dict:
        """Get summary of the active call session."""
        return {
            "total_chunks": len(self.chunks),
            "total_words": len(self.word_tokens),
            "full_transcript": self.full_transcript,
            "peak_risk_score": self.peak_risk_score,
            "peak_risk_level": self.peak_risk_level,
            "peak_categories": self.peak_categories,
            "history": self.session_history,
        }
