"""
Streaming / Partial Transcript Analyzer
Handles live call audio transcripts arriving in real-time incremental chunks.
Maintains session state, rolling windows, and cumulative evidence accumulation.
"""

from typing import Dict, List, Optional
from .ensemble_scorer import EnsembleScorer


class StreamingScamDetector:
    """
    Session-level streaming transcript analyzer for real-time call monitoring.
    """
    
    def __init__(self, window_size_words: int = 150):
        """
        Initialize streaming session detector.
        
        Args:
            window_size_words: Max rolling word window size for density calculations.
        """
        self.ensemble = EnsembleScorer()
        self.window_size_words = window_size_words
        self.reset_session()
    
    def reset_session(self):
        """Reset session state for a new call."""
        self.chunks: List[str] = []
        self.full_transcript: str = ""
        self.word_tokens: List[str] = []
        self.peak_risk_score: float = 0.0
        self.peak_risk_level: str = "LOW"
        self.session_history: List[Dict] = []
    
    def process_chunk(self, chunk_text: str) -> Dict:
        """
        Process an incoming transcript chunk from live STT stream.
        
        Args:
            chunk_text: Text snippet of new audio transcription.
            
        Returns:
            Dict containing current_chunk_score, cumulative_score, peak_score, risk_level, and flags.
        """
        chunk_clean = chunk_text.strip()
        if not chunk_clean:
            return self.get_session_summary()
        
        self.chunks.append(chunk_clean)
        self.full_transcript = " ".join(self.chunks)
        self.word_tokens = self.full_transcript.split()
        
        # Extract rolling window for density stability
        window_tokens = self.word_tokens[-self.window_size_words:]
        window_text = " ".join(window_tokens)
        
        # Analyze full cumulative transcript & rolling window
        cum_analysis = self.ensemble.analyze(self.full_transcript)
        window_analysis = self.ensemble.analyze(window_text)
        
        current_score = max(cum_analysis['ensemble_score'], window_analysis['ensemble_score'])
        current_level = cum_analysis['ensemble_level']
        
        # Track peak risk
        if current_score > self.peak_risk_score:
            self.peak_risk_score = current_score
            self.peak_risk_level = current_level
        
        chunk_result = {
            "chunk_index": len(self.chunks),
            "chunk_text": chunk_clean,
            "total_words": len(self.word_tokens),
            "cumulative_score": current_score,
            "cumulative_level": current_level,
            "peak_score": self.peak_risk_score,
            "peak_level": self.peak_risk_level,
            "needs_l2_review": cum_analysis['needs_layer2_review'],
            "categories_triggered": cum_analysis['detection']['categories_triggered'],
            "latency_ms": cum_analysis['latency_ms'],
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
            "history": self.session_history,
        }
