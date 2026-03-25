"""Quick test - print minimal results"""
from scam_detector.ensemble_scorer import EnsembleScorer
from data.sample_scripts import SCAM_SCRIPTS, LEGITIMATE_SCRIPTS

scorer = EnsembleScorer()

print("SCAM SCRIPTS:")
for name, script in list(SCAM_SCRIPTS.items())[:3]:
    a = scorer.analyze(script)
    print(f"  {name}: {a['ensemble_score']:.0f}/100 [{a['ensemble_level']}]")

print("\nLEGIT SCRIPTS:")
for name, script in list(LEGITIMATE_SCRIPTS.items())[:3]:
    a = scorer.analyze(script)
    print(f"  {name}: {a['ensemble_score']:.0f}/100 [{a['ensemble_level']}]")

print("\nDONE!")
