"""
Simulate live 500ms audio stream delivery to verify chunking and low-latency interception.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np
from scam_detector.audio_chunker import AudioStreamChunker
from scam_detector.streaming import StreamingScamDetector

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

def test_live_stream():
    chunker = AudioStreamChunker()
    detector = StreamingScamDetector(window_size_words=150)

    print("=" * 65)
    print("  SIMULATING LIVE AUDIO STREAM (500ms Chunks @ 16kHz)")
    print("=" * 65)

    # Generate 500ms synthetic speech frames (sine wave with voice energy)
    t = np.linspace(0, 0.5, 8000, False)
    synthetic_speech_frame = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    for step in range(8):
        time.sleep(0.05)  # Simulate stream tick
        audio_window = chunker.add_audio_frame(synthetic_speech_frame)

        if audio_window is not None:
            raw_text = mock_asr_engine(audio_window)
            new_words = chunker.deduplicate_transcript(raw_text)

            if new_words:
                result = detector.process_chunk(new_words)
                bar = "█" * int(result["cumulative_score"] / 5)
                print(f"⏱️  Chunk {step+1:02d} | Emitted: \"{new_words}\"")
                print(f"    ↳ Score: [{bar:<20}] {result['cumulative_score']:5.1f}/100 | Level: {result['cumulative_level']} | Latency: {result['latency_ms']:.2f}ms\n")

if __name__ == "__main__":
    test_live_stream()
