#!/usr/bin/env python3
"""
Download Piper TTS voice models — runs fully locally, no API key needed.

Plain language: Piper is the program that turns AISHA's text replies into
speech on your laptop (CPU only, no cloud). It needs a voice file
(.onnx) + a config file (.onnx.json) to work. This script downloads
those files once and saves them so aisha_web.py can find them later.

Where files go: ~/.local/share/piper_tts/  (Linux/macOS/Windows home)
  Fallback also checked: ./models/piper/  and  ./piper_tts/

Voice chosen: en_US-ryan-medium — clear US English, medium quality.
  - ~60 MB download, CPU-friendly (onnxruntime CPU provider)
  - CPU-only: we never use CUDA here. Piper runs on CPU by design.

If this script fails, check: internet access for the one-time download,
and that you have ~100 MB free disk.
"""

import os
import sys
import requests
from pathlib import Path

# Where Piper looks for voices (checked by aisha_web.py find_piper_model)
piper_dir = Path.home() / ".local" / "share" / "piper_tts"
# Also create a local fallback inside the project
local_dir = Path(__file__).parent / "models" / "piper"

for d in (piper_dir, local_dir):
    d.mkdir(parents=True, exist_ok=True)

# Hugging Face URL for Piper voices. The config file is named .onnx.json,
# not just .json — this is a common gotcha (we handle both on load).
models = [
    (
        "en_US-ryan-medium",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium",
    ),
]

print("Downloading Piper voice models (one-time, ~60-100 MB)...")
print(f"Primary location: {piper_dir}")
print(f"Fallback location: {local_dir}")

ok_all = True

for model_name, base_url in models:
    # Real files on Hugging Face:
    #   en_US-ryan-medium.onnx
    #   en_US-ryan-medium.onnx.json   <- note the .onnx.json suffix
    config_url = f"{base_url}/{model_name}.onnx.json"
    model_url = f"{base_url}/{model_name}.onnx"

    # Save as .onnx + .onnx.json (what PiperVoice.load expects when given
    # model_path + config_path). We also keep a .json copy for compatibility.
    config_path = piper_dir / f"{model_name}.onnx.json"
    model_path = piper_dir / f"{model_name}.onnx"
    # Legacy/fallback name (some older scripts look for this)
    config_path_legacy = piper_dir / f"{model_name}.json"

    print(f"\n{model_name}:")

    # --- Config ---
    if config_path.exists():
        print(f"  ✓ Config exists: {config_path.name} ({config_path.stat().st_size/1024:.1f} KB)")
    elif config_path_legacy.exists():
        print(f"  ✓ Config exists (legacy name): {config_path_legacy.name}")
        # Also write the .onnx.json copy so Piper finds it without guessing
        try:
            config_path.write_bytes(config_path_legacy.read_bytes())
            print(f"    -> copied to {config_path.name}")
        except Exception as e:
            print(f"    -> copy failed: {e}")
    else:
        print(f"  Downloading config... ", end="", flush=True)
        try:
            r = requests.get(config_url, timeout=60)
            if r.status_code == 200:
                config_path.write_bytes(r.content)
                # Also write legacy copy
                try:
                    config_path_legacy.write_bytes(r.content)
                except Exception:
                    pass
                print(f"✓ ({len(r.content)/1024:.1f} KB)")
            else:
                print(f"✗ HTTP {r.status_code} — URL: {config_url}")
                ok_all = False
        except Exception as e:
            print(f"✗ {e}")
            ok_all = False

    # --- Model ---
    if model_path.exists():
        print(f"  ✓ Model exists: {model_path.name} ({model_path.stat().st_size/1024/1024:.1f} MB)")
    else:
        print(f"  Downloading model (~60 MB)... ", end="", flush=True)
        try:
            r = requests.get(model_url, stream=True, timeout=180)
            if r.status_code == 200:
                total = int(r.headers.get("content-length", 0))
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                size_mb = model_path.stat().st_size / 1024 / 1024
                # Copy to local fallback as well
                try:
                    fallback = local_dir / model_path.name
                    if not fallback.exists():
                        fallback.write_bytes(model_path.read_bytes())
                    cfg_fallback = local_dir / config_path.name
                    if not cfg_fallback.exists() and config_path.exists():
                        cfg_fallback.write_bytes(config_path.read_bytes())
                except Exception:
                    pass
                print(f"✓ ({size_mb:.1f} MB)")
                if total and abs(model_path.stat().st_size - total) > 1024:
                    print(f"    Note: expected {total/1024/1024:.1f} MB, got {size_mb:.1f} MB (incomplete download?)")
            else:
                print(f"✗ HTTP {r.status_code}")
                ok_all = False
        except Exception as e:
            print(f"✗ {e}")
            ok_all = False

print("\n" + ("=" * 60))
if ok_all and (piper_dir / "en_US-ryan-medium.onnx").exists():
    print(f"✓ Models ready at: {piper_dir}")
    print("  aisha_web.py will find them automatically on next start.")
else:
    print(f"Models location: {piper_dir}")
    if not ok_all:
        print("⚠ Some downloads failed — check internet and try again:")
        print("  python download_piper_models.py")
        sys.exit(1)
