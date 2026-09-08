#!/usr/bin/env python3
"""
Download model weights for Sentinel-AI Tier 2 SLM Judge.

Supported models:
  1. Gemma3-1B-IT (LiteRT-LM)  — .litertlm from litert-community/Gemma3-1B-IT (~700 MB)
  2. Qwen2.5-0.5B-Instruct (GGUF) — .gguf from Qwen/Qwen2.5-0.5B-Instruct-GGUF (~350 MB)

Usage:
    python scripts/download_weights.py              # Downloads both
    python scripts/download_weights.py --litert     # Gemma3 LiteRT only
    python scripts/download_weights.py --gguf       # Qwen GGUF only

After download, set environment variables or pass paths directly:
    export SENTINEL_LITERT_PATH=models/gemma3-1b-it.litertlm
    export SENTINEL_GGUF_PATH=models/qwen2.5-0.5b-instruct-q4_k_m.gguf
"""

import argparse
import glob
import os
import sys


# ── Configuration ───────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

LITERT_REPO = "litert-community/Gemma3-1B-IT"
LITERT_LOCAL_NAME = "gemma3-1b-it.litertlm"

GGUF_MODELS = {
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "description": "Qwen2.5-0.5B-Instruct Q4_K_M (~350 MB) — legacy GGUF backend",
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
                        bar = "=" * filled + "-" * (bar_len - filled)
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


def download_litert_model():
    """Download Gemma3-1B-IT .litertlm from HuggingFace via huggingface_hub."""
    dest_path = os.path.join(MODELS_DIR, LITERT_LOCAL_NAME)

    print(f"\n  Model: {LITERT_LOCAL_NAME}")
    print(f"  Repo:  {LITERT_REPO}")
    print(f"  Info:  Gemma3-1B-IT LiteRT-LM — Google edge runtime")

    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  [SKIP] Already exists ({size_mb:.1f} MB), skipping.")
        return True

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("  [X] huggingface_hub not installed. Install with:")
        print("       pip install huggingface_hub>=0.20.0")
        return False

    try:
        # Find the .litertlm file in the repo
        print(f"  Scanning repo {LITERT_REPO} for .litertlm files...")
        repo_files = list_repo_files(LITERT_REPO)
        litert_files = [f for f in repo_files if f.endswith(".litertlm")]

        if not litert_files:
            print(f"  [X] No .litertlm files found in {LITERT_REPO}")
            print(f"  Available files: {', '.join(repo_files[:10])}")
            return False

        # Use the first .litertlm file found
        remote_filename = litert_files[0]
        print(f"  Found: {remote_filename}")
        print(f"  Downloading via huggingface_hub (resumable)...")

        downloaded_path = hf_hub_download(
            repo_id=LITERT_REPO,
            filename=remote_filename,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
        )

        # Rename to our standard name if needed
        if os.path.basename(downloaded_path) != LITERT_LOCAL_NAME:
            final_path = os.path.join(MODELS_DIR, LITERT_LOCAL_NAME)
            os.replace(downloaded_path, final_path)
            print(f"  Renamed → {LITERT_LOCAL_NAME}")
            downloaded_path = final_path

        size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
        print(f"  [OK] Saved: {downloaded_path} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  [X] Download failed: {e}")
        return False


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
        print(f"  [OK] Saved: {downloaded}")
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  huggingface_hub download failed ({e}), falling back to urllib...")
        return False


def download_gguf_models():
    """Download GGUF model weights."""
    for filename, info in GGUF_MODELS.items():
        dest_path = os.path.join(MODELS_DIR, filename)

        print(f"\n  Model: {filename}")
        print(f"  Info:  {info['description']}")

        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  [SKIP] Already exists ({size_mb:.1f} MB), skipping.")
            continue

        # Try huggingface_hub first (resumable), fall back to urllib
        if not try_huggingface_hub_download(filename, info, dest_path):
            download_with_progress(info["url"], dest_path)


def main():
    parser = argparse.ArgumentParser(description="Download Sentinel-AI Tier 2 SLM weights")
    parser.add_argument("--litert", action="store_true", help="Download Gemma3-1B-IT LiteRT model only")
    parser.add_argument("--gguf", action="store_true", help="Download Qwen GGUF model only")
    args = parser.parse_args()

    # If neither flag is set, download both
    download_all = not args.litert and not args.gguf

    print("=" * 60)
    print("  Sentinel-AI: Download Tier 2 SLM Weights")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    if download_all or args.litert:
        download_litert_model()

    if download_all or args.gguf:
        download_gguf_models()

    print("\n" + "=" * 60)
    print("  Download complete!")
    print(f"\n  LiteRT-LM (recommended):")
    print(f"    export SENTINEL_LITERT_PATH={MODELS_DIR}/gemma3-1b-it.litertlm")
    print(f"\n  GGUF (legacy):")
    print(f"    export SENTINEL_GGUF_PATH={MODELS_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    print("=" * 60)


if __name__ == "__main__":
    main()
