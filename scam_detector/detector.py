"""
Main Detection Engine
Performs keyword matching, pattern analysis, and word tokenization.
Includes high-precision latency instrumentation (latency_ms).
"""

import time
import re
from typing import Dict, List, Set, Tuple
from .word_dictionary import SCAM_WORDS, CATEGORY_WEIGHTS
from .patterns import PATTERNS


class ScamDetector:
    """
    Main detection engine for scam call analysis.
    Identifies suspicious keywords, regex patterns, and density metrics.
    """
    
    def __init__(self):
        self.scam_words = SCAM_WORDS
        self.patterns = PATTERNS
        self.category_weights = CATEGORY_WEIGHTS
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        cleaned = self.clean_text(text)
        return cleaned.split()
    
    def find_suspicious_words(self, text: str) -> Tuple[Dict[str, List[Tuple[str, int]]], Set[str]]:
        """
        Find suspicious words and phrases by category.
        
        Returns:
            Tuple of (detected_words_by_category, set_of_all_matched_keywords)
        """
        cleaned_text = f" {self.clean_text(text)} "
        tokens = self.tokenize(text)
        
        detected_words: Dict[str, List[Tuple[str, int]]] = {}
        all_matched_keywords: Set[str] = set()
        
        for category, word_set in self.scam_words.items():
            category_matches = []
            
            for word in word_set:
                word_lower = word.lower()
                
                if " " in word_lower:
                    # Multi-word phrase matching
                    if f" {word_lower} " in cleaned_text:
                        category_matches.append((word_lower, 0))
                        all_matched_keywords.add(word_lower)
                else:
                    # Single word token matching
                    for idx, token in enumerate(tokens):
                        if token == word_lower:
                            category_matches.append((word_lower, idx))
                            all_matched_keywords.add(word_lower)
            
            if category_matches:
                detected_words[category] = category_matches
        
        return detected_words, all_matched_keywords
    
    def find_patterns(self, text: str) -> Dict[str, List[str]]:
        """Find regex pattern matches in text."""
        detected_patterns: Dict[str, List[str]] = {}
        
        for pattern_name, regex_list in self.patterns.items():
            matches = []
            for regex in regex_list:
                found = regex.findall(text)
                for match in found:
                    if isinstance(match, tuple):
                        match_str = " ".join([m for m in match if m])
                    else:
                        match_str = match
                    if match_str and match_str not in matches:
                        matches.append(match_str)
            
            if matches:
                detected_patterns[pattern_name] = matches
        
        return detected_patterns
    
    def analyze(self, text: str) -> Dict:
        """
        Perform complete scam detection analysis on input text.
        
        Returns:
            Analysis dictionary with categories, matched words, patterns, metrics, and latency_ms.
        """
        t0 = time.perf_counter()
        
        tokens = self.tokenize(text)
        word_count = len(tokens)
        
        detected_words, matched_keywords = self.find_suspicious_words(text)
        detected_patterns = self.find_patterns(text)
        
        categories_triggered = list(detected_words.keys())
        
        # Total suspicious word hits
        suspicious_word_count = sum(len(words) for words in detected_words.values())
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        return {
            "text": text,
            "word_count": word_count,
            "suspicious_word_count": suspicious_word_count,
            "detected_words": detected_words,
            "matched_keywords": list(matched_keywords),
            "detected_patterns": detected_patterns,
            "categories_triggered": categories_triggered,
            "latency_ms": elapsed_ms,
        }
    
    def get_detection_summary(self, analysis: Dict) -> str:
        """Generate human-readable summary of detection analysis."""
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("DETECTION SUMMARY")
        lines.append("=" * 60)
        
        lines.append(f"\n📊 Total Words: {analysis['word_count']}")
        lines.append(f"🚩 Suspicious Hits: {analysis['suspicious_word_count']}")
        lines.append(f"⚡ Latency: {analysis['latency_ms']:.2f} ms")
        
        if analysis['categories_triggered']:
            lines.append(f"\n🎯 Categories Triggered ({len(analysis['categories_triggered'])}):")
            for cat in analysis['categories_triggered']:
                words = [w for w, _ in analysis['detected_words'][cat]]
                lines.append(f"   • {cat:20} -> {', '.join(set(words))}")
        else:
            lines.append("\n✅ No suspicious categories triggered.")
        
        if analysis['detected_patterns']:
            lines.append(f"\n🔍 Regex Patterns Matched ({len(analysis['detected_patterns'])}):")
            for pattern, matches in analysis['detected_patterns'].items():
                lines.append(f"   • {pattern:25} -> {', '.join(matches[:2])}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
