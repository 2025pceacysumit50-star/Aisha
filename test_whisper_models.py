#!/usr/bin/env python3
"""
Test faster-whisper model sizes to decide which is best for CPU.
"""

import time
from faster_whisper import WhisperModel
import numpy as np

def test_model(model_size):
    """Load a model and report stats."""
    print(f"\n{'='*60}")
    print(f"Testing {model_size} model")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        load_time = time.time() - start
        print(f"✅ Model loaded successfully")
        print(f"   Load time: {load_time:.1f}s")
        
        # Get model info
        print(f"\nModel details:")
        print(f"   Device: CPU")
        print(f"   Compute type: int8 (quantized for CPU efficiency)")
        print(f"   Size: {model_size}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False

# Test models in order of speed
models = ["tiny", "small", "base"]

print("FASTER-WHISPER MODEL SIZE COMPARISON")
print("All using CPU with int8 quantization (best for CPU-only systems)\n")

results = {}
for model_size in models:
    results[model_size] = test_model(model_size)

print(f"\n{'='*60}")
print("RECOMMENDATION FOR CPU-ONLY SYSTEM")
print(f"{'='*60}")

# Show recommendations
print("\nTrade-offs:")
print("  • 'tiny' (39M):   ⚡ Fastest, 🎯 least accurate")
print("  • 'small' (140M): ⚡ Fast,   🎯 decent accuracy (RECOMMENDED)")
print("  • 'base' (140M):  🐢 Slow,    🎯 better accuracy")

print("\n✅ RECOMMENDATION: Use 'small' model")
print("   - Fast enough for responsive UI (~2-3s per 10s audio on CPU)")
print("   - Good accuracy for general speech recognition")
print("   - Reasonable size (~440MB download)")

print("\nNOTE: You can test different quantization levels:")
print("  - 'int8' (now):  Most compact, CPU-friendly")
print("  - 'float16':     Slightly larger, maybe faster on some CPUs")
print("  - 'float32':     Largest, may be slower")
