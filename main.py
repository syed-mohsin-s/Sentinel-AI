"""
Scam Call Detection System - CLI Interface
Main entry point for testing and analyzing text.
"""

import sys
from scam_detector import ScamDetector, RiskScorer
from data.sample_scripts import get_all_scripts, get_script_by_name


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║           🔍 SCAM CALL DETECTION SYSTEM v1.0 🔍              ║
║         Detect suspicious words and scam patterns            ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def analyze_text(text: str, title: str = "Input Text"):
    """Analyze text and print results."""
    detector = ScamDetector()
    scorer = RiskScorer()
    
    print(f"\n{'='*60}")
    print(f"📝 Analyzing: {title}")
    print(f"{'='*60}")
    
    # Show preview of text
    preview = text[:200].strip().replace('\n', ' ')
    if len(text) > 200:
        preview += "..."
    print(f"\nText Preview: \"{preview}\"")
    
    # Run analysis
    analysis = detector.analyze(text)
    
    # Print detection results
    print(detector.get_detection_summary(analysis))
    
    # Print risk score
    print(scorer.get_score_report(analysis))
    
    return analysis


def run_demo():
    """Run demo with sample scripts."""
    print("\n" + "="*60)
    print("🎭 DEMO MODE - Testing with Sample Scripts")
    print("="*60)
    
    all_scripts = get_all_scripts()
    
    # Test scam scripts
    print("\n\n" + "🚨 "*20)
    print("TESTING SCAM SCRIPTS (should show HIGH/CRITICAL risk)")
    print("🚨 "*20)
    
    for name, script in all_scripts["scam"].items():
        analyze_text(script, f"SCAM: {name}")
        print("\n" + "-"*60)
    
    # Test legitimate scripts
    print("\n\n" + "✅ "*20)
    print("TESTING LEGITIMATE SCRIPTS (should show LOW risk)")
    print("✅ "*20)
    
    for name, script in all_scripts["legitimate"].items():
        analyze_text(script, f"LEGITIMATE: {name}")
        print("\n" + "-"*60)
    
    # Test borderline scripts
    print("\n\n" + "⚠️ "*20)
    print("TESTING BORDERLINE SCRIPTS (may show MEDIUM risk)")
    print("⚠️ "*20)
    
    for name, script in all_scripts["borderline"].items():
        analyze_text(script, f"BORDERLINE: {name}")
        print("\n" + "-"*60)


def interactive_mode():
    """Run interactive text analysis mode."""
    print("\n" + "="*60)
    print("💬 INTERACTIVE MODE")
    print("Enter text to analyze (type 'quit' to exit, 'demo' for demo)")
    print("="*60)
    
    while True:
        print("\n" + "-"*40)
        print("Enter text (or paste multi-line, then press Enter twice):")
        print("Commands: 'quit', 'demo', 'script:<name>'")
        print("-"*40)
        
        lines = []
        while True:
            try:
                line = input()
                if line == "":
                    if lines:
                        break
                else:
                    lines.append(line)
            except EOFError:
                break
        
        text = "\n".join(lines).strip()
        
        if not text:
            continue
        
        if text.lower() == "quit":
            print("\n👋 Goodbye!")
            break
        
        if text.lower() == "demo":
            run_demo()
            continue
        
        if text.lower().startswith("script:"):
            script_name = text.split(":", 1)[1].strip()
            script = get_script_by_name(script_name)
            if script:
                analyze_text(script, f"Script: {script_name}")
            else:
                print(f"❌ Script '{script_name}' not found.")
                print("Available scripts:")
                all_scripts = get_all_scripts()
                for category, scripts in all_scripts.items():
                    print(f"  {category}: {', '.join(scripts.keys())}")
            continue
        
        analyze_text(text, "User Input")


def main():
    """Main entry point."""
    print_banner()
    
    if len(sys.argv) > 1:
        # Command line arguments
        if sys.argv[1] == "--demo":
            run_demo()
        elif sys.argv[1] == "--file":
            if len(sys.argv) > 2:
                try:
                    with open(sys.argv[2], 'r', encoding='utf-8') as f:
                        text = f.read()
                    analyze_text(text, f"File: {sys.argv[2]}")
                except FileNotFoundError:
                    print(f"❌ File not found: {sys.argv[2]}")
                except Exception as e:
                    print(f"❌ Error reading file: {e}")
            else:
                print("Usage: python main.py --file <path>")
        elif sys.argv[1] == "--text":
            if len(sys.argv) > 2:
                text = " ".join(sys.argv[2:])
                analyze_text(text, "Command Line Input")
            else:
                print("Usage: python main.py --text <text>")
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python main.py              # Interactive mode")
            print("  python main.py --demo       # Run demo with sample scripts")
            print("  python main.py --file PATH  # Analyze text file")
            print("  python main.py --text TEXT  # Analyze provided text")
        else:
            # Treat as text input
            text = " ".join(sys.argv[1:])
            analyze_text(text, "Command Line Input")
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
