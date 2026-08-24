"""
Simple Layer 1 Test - Summary View
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scam_detector.ensemble_scorer import EnsembleScorer
from data.sample_scripts import SCAM_SCRIPTS, LEGITIMATE_SCRIPTS, BORDERLINE_SCRIPTS

def main():
    print("=" * 80)
    print("  LAYER 1 ENSEMBLE TEST (40% Keyword + 60% Semantic)")
    print("=" * 80)
    
    scorer = EnsembleScorer()
    
    all_tests = [
        ("🚨 SCAM SCRIPTS (Expected: HIGH/CRITICAL)", SCAM_SCRIPTS),
        ("✅ LEGITIMATE SCRIPTS (Expected: LOW)", LEGITIMATE_SCRIPTS),
        ("⚠️ BORDERLINE / TELEMARKETING (Expected: MEDIUM/LOW)", BORDERLINE_SCRIPTS),
    ]
    
    for category_name, scripts in all_tests:
        print(f"\n{category_name}")
        print("-" * 80)
        
        for name, script in scripts.items():
            analysis = scorer.analyze(script)
            
            score = analysis['ensemble_score']
            level = analysis['ensemble_level']
            kw_score = analysis['keyword_score']
            sem_score = analysis['semantic_score']
            latency = analysis['latency_ms']
            needs_l2 = "[L2]" if analysis['needs_layer2_review'] else "    "
            
            bar_len = 20
            filled = int((score / 100) * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            print(f"  {name:28} [{bar}] {score:5.1f} | KW:{kw_score:4.0f} SEM:{sem_score:4.0f} | {level:8} {needs_l2} ({latency:.2f}ms)")
    
    print("\n" + "=" * 80)
    print("Legend: KW=Keyword Score, SEM=Semantic Score, [L2]=Needs Layer 2 Review")
    print("=" * 80)

if __name__ == "__main__":
    main()
