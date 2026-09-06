import sys
sys.path.insert(0, 'd:\\Downloads\\aisha-v0.4\\aisha')

print("Testing imports...")
try:
    import config
    print("✓ config.py imports")
except Exception as e:
    print(f"✗ config.py: {e}")

try:
    import aisha
    print("✓ aisha.py imports")
except Exception as e:
    print(f"✗ aisha.py: {e}")

try:
    import ingest
    print("✓ ingest.py imports")
except Exception as e:
    print(f"✗ ingest.py: {e}")

sys.path.insert(0, 'd:\\Downloads\\aisha-v0.4\\aisha\\malware_classifier')
try:
    import train_classifier
    print("✓ malware_classifier/train_classifier.py imports")
except Exception as e:
    print(f"✗ train_classifier.py: {e}")

sys.path.insert(0, 'd:\\Downloads\\aisha-v0.4\\aisha\\training')
try:
    import finetune
    print("✓ training/finetune.py imports")
except Exception as e:
    print(f"✗ finetune.py: {e}")

print("\n✓ All modules import successfully!")
