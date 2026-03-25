"""
Test Layer 1: Keyword + TinyBERT Ensemble
"""

from scam_detector.ensemble_scorer import EnsembleScorer, quick_analyze
from data.sample_scripts import SCAM_SCRIPTS, LEGITIMATE_SCRIPTS, BORDERLINE_SCRIPTS

def run_tests():
    print("=" * 65)
    print("  LAYER 1 ENSEMBLE TEST: Keywords + TinyBERT")
    print("=" * 65)
    
    scorer = EnsembleScorer()
    
    results = {
        "scam": [],
        "legitimate": [],
        "borderline": []
    }
    
    # Test Scam Scripts
    print("\n" + "=" * 65)
    print("  TESTING SCAM SCRIPTS (Expected: HIGH/CRITICAL)")
    print("=" * 65)
    
    for name, script in SCAM_SCRIPTS.items():
        print(f"\n>>> Testing: {name}")
        analysis = scorer.analyze(script)
        results["scam"].append({
            "name": name,
            "score": analysis['ensemble_score'],
            "level": analysis['ensemble_level'],
            "needs_l2": analysis['needs_layer2_review']
        })
        print(scorer.get_report(analysis))
    
    # Test Legitimate Scripts
    print("\n" + "=" * 65)
    print("  TESTING LEGITIMATE SCRIPTS (Expected: LOW)")
    print("=" * 65)
    
    for name, script in LEGITIMATE_SCRIPTS.items():
        print(f"\n>>> Testing: {name}")
        analysis = scorer.analyze(script)
        results["legitimate"].append({
            "name": name,
            "score": analysis['ensemble_score'],
            "level": analysis['ensemble_level'],
            "needs_l2": analysis['needs_layer2_review']
        })
        print(scorer.get_report(analysis))
    
    # Test Borderline Scripts
    print("\n" + "=" * 65)
    print("  TESTING BORDERLINE SCRIPTS (Expected: MEDIUM)")
    print("=" * 65)
    
    for name, script in BORDERLINE_SCRIPTS.items():
        print(f"\n>>> Testing: {name}")
        analysis = scorer.analyze(script)
        results["borderline"].append({
            "name": name,
            "score": analysis['ensemble_score'],
            "level": analysis['ensemble_level'],
            "needs_l2": analysis['needs_layer2_review']
        })
        print(scorer.get_report(analysis))
    
    # Summary
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    
    print("\nSCAM SCRIPTS:")
    for r in results["scam"]:
        l2_flag = "[L2]" if r['needs_l2'] else ""
        print(f"  {r['name']:25} -> {r['score']:5.1f} [{r['level']:8}] {l2_flag}")
    
    print("\nLEGITIMATE SCRIPTS:")
    for r in results["legitimate"]:
        l2_flag = "[L2]" if r['needs_l2'] else ""
        print(f"  {r['name']:25} -> {r['score']:5.1f} [{r['level']:8}] {l2_flag}")
    
    print("\nBORDERLINE SCRIPTS:")
    for r in results["borderline"]:
        l2_flag = "[L2]" if r['needs_l2'] else ""
        print(f"  {r['name']:25} -> {r['score']:5.1f} [{r['level']:8}] {l2_flag}")
    
    # Accuracy check
    scam_correct = sum(1 for r in results["scam"] if r['level'] in ['HIGH', 'CRITICAL'])
    legit_correct = sum(1 for r in results["legitimate"] if r['level'] == 'LOW')
    
    print(f"\n--- Accuracy ---")
    print(f"Scam Detection: {scam_correct}/{len(results['scam'])} correct")
    print(f"Legit Detection: {legit_correct}/{len(results['legitimate'])} correct")
    
    print("\n" + "=" * 65)
    print("  TEST COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
