#!/usr/bin/env python3
"""
Download GGUF model weights for Sentinel-AI Tier 2 SLM Judge.

Downloads a quantized Qwen2.5-0.5B-Instruct GGUF from Hugging Face Hub
into the local models/ directory.  The file is ~350 MB.

Usage:
    python scripts/download_weights.py

The downloaded model can be used by setting:
    export SENTINEL_GGUF_PATH=models/qwen2.5-0.5b-instruct-q4_k_m.gguf

Or by passing model_path directly:
    Tier2SLMJudge(model_path="models/qwen2.5-0.5b-instruct-q4_k_m.gguf", backend="llama_cpp")
"""

import os
import sys
import hashlib

# ── Configuration ───────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

MODELS = {
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "description": "Qwen2.5-0.5B-Instruct Q4_K_M (~350 MB) — recommended default",
    },
}


def download_with_progress(url: str, dest_path: str):
    """Download a file with a progress bar using only stdlib."""
    import urllib.request

    print(f"  Downloading: {url}")
    print(f"  Destination: {dest_path}")

    tmp_path = dest_path + ".part"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Sentinel-AI/2.0"})
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = downloaded / total * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        bar_len = 30
                        filled = int(bar_len * downloaded / total)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        print(f"\r  [{bar}] {pct:5.1f}%  ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        print(f"\r  Downloaded {mb_done:.1f} MB...", end="", flush=True)

        print()  # newline after progress bar

        # Rename .part to final name atomically
        os.replace(tmp_path, dest_path)
        print(f"  ✅ Saved: {dest_path} ({os.path.getsize(dest_path) / (1024*1024):.1f} MB)")

    except Exception as e:
        # Clean up partial download
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Download failed: {e}") from e


def try_huggingface_hub_download(filename: str, model_info: dict, dest_path: str) -> bool:
    """Attempt download via huggingface_hub if available (supports resumable downloads)."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"  Using huggingface_hub for resumable download...")
        downloaded = hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename=filename,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"  ✅ Saved: {downloaded}")
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  huggingface_hub download failed ({e}), falling back to urllib...")
        return False


def main():
    print("=" * 60)
    print("  Sentinel-AI: Download Tier 2 SLM Weights")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, info in MODELS.items():
        dest_path = os.path.join(MODELS_DIR, filename)

        print(f"\n  Model: {filename}")
        print(f"  Info:  {info['description']}")

        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  ⏭️  Already exists ({size_mb:.1f} MB), skipping.")
            continue

        # Try huggingface_hub first (resumable), fall back to urllib
        if not try_huggingface_hub_download(filename, info, dest_path):
            download_with_progress(info["url"], dest_path)

    print("\n" + "=" * 60)
    print("  Download complete!")
    print(f"  Set SENTINEL_GGUF_PATH to use with Tier2SLMJudge:")
    print(f"    export SENTINEL_GGUF_PATH={MODELS_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    print("=" * 60)


if __name__ == "__main__":
    main()
