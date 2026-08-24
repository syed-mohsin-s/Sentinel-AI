"""
Test Layer 1: Keyword + Semantic Ensemble
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scam_detector.ensemble_scorer import EnsembleScorer, quick_analyze
from data.sample_scripts import SCAM_SCRIPTS, LEGITIMATE_SCRIPTS, BORDERLINE_SCRIPTS

def run_tests():
    print("=" * 65)
    print("  LAYER 1 ENSEMBLE TEST: Keywords + Semantic Layer")
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
            "needs_l2": analysis['needs_layer2_review'],
            "latency_ms": analysis['latency_ms']
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
            "needs_l2": analysis['needs_layer2_review'],
            "latency_ms": analysis['latency_ms']
        })
        print(scorer.get_report(analysis))
    
    # Test Borderline Scripts
    print("\n" + "=" * 65)
    print("  TESTING BORDERLINE SCRIPTS (Expected: MEDIUM / LOW)")
    print("=" * 65)
    
    for name, script in BORDERLINE_SCRIPTS.items():
        print(f"\n>>> Testing: {name}")
        analysis = scorer.analyze(script)
        results["borderline"].append({
            "name": name,
            "score": analysis['ensemble_score'],
            "level": analysis['ensemble_level'],
            "needs_l2": analysis['needs_layer2_review'],
            "latency_ms": analysis['latency_ms']
        })
        print(scorer.get_report(analysis))
    
    # Summary
    print("\n" + "=" * 65)
    print("  SUMMARY REPORT")
    print("=" * 65)
    
    print("\nSCAM SCRIPTS:")
    for r in results["scam"]:
        l2_flag = "[L2]" if r['needs_l2'] else ""
        print(f"  {r['name']:28} -> {r['score']:5.1f} [{r['level']:8}] {l2_flag:5} ({r['latency_ms']:.2f}ms)")
    
    print("\nLEGITIMATE SCRIPTS:")
    for r in results["legitimate"]:
        l2_flag = "[L2]" if r['needs_l2'] else ""
        print(f"  {r['name']:28} -> {r['score']:5.1f} [{r['level']:8}] {l2_flag:5} ({r['latency_ms']:.2f}ms)")
    
    print("\nBORDERLINE / TELEMARKETING SCRIPTS:")
    for r in results["borderline"]:
        l2_flag = "[L2]" if r['needs_l2'] else ""
        print(f"  {r['name']:28} -> {r['score']:5.1f} [{r['level']:8}] {l2_flag:5} ({r['latency_ms']:.2f}ms)")
    
    # Accuracy check
    scam_correct = sum(1 for r in results["scam"] if r['level'] in ['HIGH', 'CRITICAL'])
    legit_correct = sum(1 for r in results["legitimate"] if r['level'] in ['LOW', 'MEDIUM'])
    borderline_safe = sum(1 for r in results["borderline"] if r['level'] in ['LOW', 'MEDIUM'])
    
    total_latency = sum(r['latency_ms'] for category in results.values() for r in category)
    total_tests = sum(len(category) for category in results.values())
    avg_latency = total_latency / total_tests if total_tests > 0 else 0
    
    print(f"\n--- Performance & Accuracy Metrics ---")
    print(f"Scam Threat Detection: {scam_correct}/{len(results['scam'])} correct")
    print(f"Legit Call Accuracy:    {legit_correct}/{len(results['legitimate'])} correct")
    print(f"Sales/Survey FP Control: {borderline_safe}/{len(results['borderline'])} safe (Non-Critical)")
    print(f"Average Inference Speed: {avg_latency:.2f} ms / call")
    
    print("\n" + "=" * 65)
    print("  TEST COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
