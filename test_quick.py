"""Quick test script for the scam detection system - ASCII safe."""
from scam_detector import ScamDetector, RiskScorer
from data.sample_scripts import SCAM_SCRIPTS, LEGITIMATE_SCRIPTS

detector = ScamDetector()
scorer = RiskScorer()

print("=" * 60)
print("SCAM DETECTION SYSTEM - VERIFICATION TEST")
print("=" * 60)

# Test scam scripts
print("\n[SCAM SCRIPTS]:")
for name, script in SCAM_SCRIPTS.items():
    analysis = detector.analyze(script)
    raw, norm, level = scorer.calculate_score(analysis)
    print(f"  [{level:8}] {name}: {norm:.0f}/100 - {analysis['suspicious_word_count']} suspicious words, {len(analysis['categories_triggered'])} categories")

# Test legitimate scripts  
print("\n[LEGITIMATE SCRIPTS]:")
for name, script in LEGITIMATE_SCRIPTS.items():
    analysis = detector.analyze(script)
    raw, norm, level = scorer.calculate_score(analysis)
    print(f"  [{level:8}] {name}: {norm:.0f}/100 - {analysis['suspicious_word_count']} suspicious words, {len(analysis['categories_triggered'])} categories")

print("\n" + "=" * 60)
print("All tests completed successfully!")
