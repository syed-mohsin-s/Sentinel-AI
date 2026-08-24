"""Quick test - print minimal results"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scam_detector.ensemble_scorer import EnsembleScorer
from data.sample_scripts import SCAM_SCRIPTS, LEGITIMATE_SCRIPTS

scorer = EnsembleScorer()

print("SCAM SCRIPTS:")
for name, script in list(SCAM_SCRIPTS.items())[:3]:
    a = scorer.analyze(script)
    print(f"  {name}: {a['ensemble_score']:.0f}/100 [{a['ensemble_level']}] ({a['latency_ms']:.2f}ms)")

print("\nLEGIT SCRIPTS:")
for name, script in list(LEGITIMATE_SCRIPTS.items())[:3]:
    a = scorer.analyze(script)
    print(f"  {name}: {a['ensemble_score']:.0f}/100 [{a['ensemble_level']}] ({a['latency_ms']:.2f}ms)")

print("\nDONE!")
