"""
Simulate live 500ms audio stream delivery to verify chunking, low-latency interception,
Tier 2 Edge SLM reasoning, and HUD overlay dispatch.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np
from scam_detector.audio_chunker import LiveCallAudioInterceptor

# Mock incremental transcripts simulating an incoming Digital Arrest scam call
MOCK_ASR_STREAM = [
    "Hello this is Inspector Sharma from Delhi Cyber Crime Branch",
    "Delhi Cyber Crime Branch your Aadhaar card details were found in FedEx parcel",
    "were found in FedEx parcel containing contraband MDMA drugs and forged passports",
    "passports a digital arrest warrant has been issued by Supreme Court",
    "issued by Supreme Court transfer 50000 rupees to police clearance account immediately"
]

def mock_asr_engine(audio_window: np.ndarray, index_holder=[0]) -> str:
    """Simulates local ASR model inference."""
    idx = min(index_holder[0], len(MOCK_ASR_STREAM) - 1)
    text = MOCK_ASR_STREAM[idx]
    index_holder[0] += 1
    return text

def telemetry_logger(payload: dict):
    """Mock callback for external HUD / secondary screen telemetry."""
    # print(f"  [TELEMETRY DISPATCHED] State: {payload['hud_state']} | Risk Score: {payload['risk_score']:.1f}")
    pass

def test_live_stream():
    interceptor = LiveCallAudioInterceptor(
        asr_engine=mock_asr_engine,
        telemetry_callback=telemetry_logger,
        use_npu=True
    )

    print("=" * 65)
    print("  SIMULATING LIVE AUDIO STREAM WITH TIER 2 SLM & HUD INTERCEPTION")
    print("=" * 65)

    # Generate 500ms synthetic speech frames (sine wave with voice energy)
    t = np.linspace(0, 0.5, 8000, False)
    synthetic_speech_frame = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    for step in range(8):
        time.sleep(0.05)  # Simulate stream tick
        result = interceptor.on_audio_chunk_received(synthetic_speech_frame)

        if result:
            new_words = result["transcript_delta"]
            bar = "█" * int(result["cumulative_score"] / 5)
            print(f"⏱️  Chunk {step+1:02d} | Emitted: \"{new_words}\"")
            print(f"    ↳ Score: [{bar:<20}] {result['cumulative_score']:5.1f}/100 | Level: {result['cumulative_level']} | Latency: {result['total_e2e_latency_ms']:.2f}ms")
            
            if "tier2_slm" in result:
                slm = result["tier2_slm"]
                print(f"    🧠 [Tier 2 SLM Verdict]: Threat={slm['threat_type']} | Confidence={slm['confidence']*100:.0f}% | Action={slm['recommended_action']}")
            print()

if __name__ == "__main__":
    test_live_stream()
