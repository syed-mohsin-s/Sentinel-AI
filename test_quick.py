"""Quick test script for the scam detection system."""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scam_detector import ScamDetector, RiskScorer
from data.sample_scripts import SCAM_SCRIPTS, LEGITIMATE_SCRIPTS

detector = ScamDetector()
scorer = RiskScorer()

print("=" * 60)
print("SCAM DETECTION SYSTEM - VERIFICATION TEST")
print("=" * 60)

print("\n[SCAM SCRIPTS]:")
for name, script in SCAM_SCRIPTS.items():
    analysis = detector.analyze(script)
    raw, norm, level = scorer.calculate_score(analysis)
    print(f"  [{level:8}] {name:28}: {norm:5.1f}/100 - {analysis['suspicious_word_count']} hits, {len(analysis['categories_triggered'])} categories ({analysis['latency_ms']:.2f}ms)")

print("\n[LEGITIMATE SCRIPTS]:")
for name, script in LEGITIMATE_SCRIPTS.items():
    analysis = detector.analyze(script)
    raw, norm, level = scorer.calculate_score(analysis)
    print(f"  [{level:8}] {name:28}: {norm:5.1f}/100 - {analysis['suspicious_word_count']} hits, {len(analysis['categories_triggered'])} categories ({analysis['latency_ms']:.2f}ms)")

print("\n" + "=" * 60)
print("All quick tests completed successfully!")
