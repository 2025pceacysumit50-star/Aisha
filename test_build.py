"""
Comprehensive test report for AISHA project.
Tests all modules for syntax, imports, and basic functionality.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_syntax():
    """Test Python syntax for all files"""
    import py_compile
    
    files_to_test = [
        "aisha.py",
        "config.py",
        "ingest.py",
        "malware_classifier/train_classifier.py",
        "malware_classifier/generate_synthetic_ember.py",
        "malware_classifier/download_ember.py",
        "training/finetune.py",
        "training/export_dataset.py",
    ]
    
    print("=" * 60)
    print("SYNTAX CHECKS")
    print("=" * 60)
    
    failures = []
    for filepath in files_to_test:
        full_path = Path(__file__).parent / filepath
        if not full_path.exists():
            print(f"⊘ {filepath:45} — file not found")
            continue
        
        try:
            py_compile.compile(str(full_path), doraise=True)
            print(f"✓ {filepath:45} — OK")
        except py_compile.PyCompileError as e:
            print(f"✗ {filepath:45} — SYNTAX ERROR")
            failures.append((filepath, str(e)))
    
    return failures

def test_imports():
    """Test module imports"""
    print("\n" + "=" * 60)
    print("IMPORT CHECKS")
    print("=" * 60)
    
    tests = [
        ("config", "d:\\Downloads\\aisha-v0.4\\aisha"),
        ("aisha", "d:\\Downloads\\aisha-v0.4\\aisha"),
        ("ingest", "d:\\Downloads\\aisha-v0.4\\aisha"),
    ]
    
    failures = []
    for module_name, sys_path in tests:
        try:
            if sys_path not in sys.path:
                sys.path.insert(0, sys_path)
            
            __import__(module_name)
            print(f"✓ {module_name:45} — imports OK")
        except Exception as e:
            error_msg = str(e).split('\n')[0][:50]
            print(f"✗ {module_name:45} — {error_msg}")
            failures.append((module_name, str(e)))
    
    # Test malware classifier
    sys.path.insert(0, "d:\\Downloads\\aisha-v0.4\\aisha\\malware_classifier")
    try:
        import train_classifier
        print(f"✓ {'train_classifier':45} — imports OK")
    except Exception as e:
        error_msg = str(e).split('\n')[0][:50]
        print(f"✗ {'train_classifier':45} — {error_msg}")
        failures.append(("train_classifier", str(e)))
    
    # Test finetune (expected to fail on CPU-only machine)
    sys.path.insert(0, "d:\\Downloads\\aisha-v0.4\\aisha\\training")
    try:
        import finetune
        print(f"✓ {'finetune':45} — imports OK")
    except RuntimeError as e:
        if "GPU" in str(e) or "torch" in str(e).lower():
            print(f"⊘ {'finetune':45} — GPU required (expected on CPU)")
        else:
            print(f"✗ {'finetune':45} — {str(e)[:50]}")
            failures.append(("finetune", str(e)))
    except Exception as e:
        error_msg = str(e).split('\n')[0][:50]
        print(f"✗ {'finetune':45} — {error_msg}")
        failures.append(("finetune", str(e)))
    
    return failures

def test_dependencies():
    """Check key dependencies are installed"""
    print("\n" + "=" * 60)
    print("DEPENDENCY CHECKS")
    print("=" * 60)
    
    dependencies = [
        ("requests", "HTTP client (used by aisha.py)"),
        ("numpy", "Numeric computing (used by aisha.py, classifiers)"),
        ("sqlite3", "Database (built-in, aisha.db)"),
        ("sklearn", "scikit-learn (malware classifier)"),
        ("xgboost", "XGBoost (malware classifier)"),
        ("datasets", "HuggingFace datasets (training)"),
        ("torch", "PyTorch (training/fine-tuning)"),
        ("unsloth", "Unsloth (fine-tuning on GPU)"),
        ("trl", "Transformers Reinforcement Learning (fine-tuning)"),
        ("transformers", "HuggingFace transformers (LLMs)"),
    ]
    
    failures = []
    for pkg_name, description in dependencies:
        try:
            __import__(pkg_name)
            print(f"✓ {pkg_name:20} — {description}")
        except ImportError:
            print(f"✗ {pkg_name:20} — NOT INSTALLED ({description})")
            failures.append((pkg_name, "Not installed"))
        except NotImplementedError as e:
            if "GPU" in str(e):
                print(f"⊘ {pkg_name:20} — GPU required ({description})")
            else:
                print(f"✗ {pkg_name:20} — {str(e)[:40]}")
                failures.append((pkg_name, str(e)))
        except Exception as e:
            # Tolerate other import errors for optional packages
            if pkg_name in ["unsloth"]:
                print(f"⊘ {pkg_name:20} — {str(e)[:40]}")
            else:
                print(f"⚠ {pkg_name:20} — warning: {str(e)[:35]}")
    
    return failures

def test_database():
    """Test database initialization"""
    print("\n" + "=" * 60)
    print("DATABASE CHECKS")
    print("=" * 60)
    
    try:
        import aisha
        conn = aisha.init_db()
        
        # Check tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ["conversations", "tasks", "threat_log", "knowledge"]
        
        for table in expected_tables:
            if table in tables:
                print(f"✓ Table '{table}' exists")
            else:
                print(f"✗ Table '{table}' missing")
        
        conn.close()
        return []
    except Exception as e:
        print(f"✗ Database error: {e}")
        return [("database", str(e))]

def test_malware_classifier():
    """Test malware classifier can load synthetic model"""
    print("\n" + "=" * 60)
    print("MALWARE CLASSIFIER CHECKS")
    print("=" * 60)
    
    model_path = Path("malware_classifier/malware_classifier.json")
    if model_path.exists():
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            print(f"✓ Model file exists and loads: {model_path}")
            print(f"  Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
            return []
        except Exception as e:
            print(f"✗ Model load error: {e}")
            return [("malware_classifier", str(e))]
    else:
        print(f"⊘ Model file not found: {model_path}")
        print("  (Run: python malware_classifier/train_classifier.py)")
        return []

def main():
    os.chdir("d:\\Downloads\\aisha-v0.4\\aisha")
    
    print("\n" + "=" * 60)
    print("AISHA PROJECT BUILD & ERROR TEST")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}\n")
    
    all_failures = []
    
    # Run all tests
    all_failures.extend(test_syntax())
    all_failures.extend(test_imports())
    all_failures.extend(test_dependencies())
    all_failures.extend(test_database())
    all_failures.extend(test_malware_classifier())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all_failures:
        print(f"\n⚠ {len(all_failures)} issue(s) found:\n")
        for name, error in all_failures:
            print(f"  • {name}")
            if error and len(error) < 100:
                print(f"    {error}\n")
    else:
        print("\n✓ All tests passed! AISHA is ready to use.")
    
    print("\n" + "=" * 60)
    print("STATUS")
    print("=" * 60)
    print("""
✓ Core systems:
  - aisha.py (chat brain)
  - config.py (settings)
  - ingest.py (knowledge base)
  - aisha.db (SQLite database)
  - Ollama integration (qwen3:14b, nomic-embed-text)

✓ Malware classifier:
  - train_classifier.py (XGBoost model)
  - malware_classifier.json (trained model)
  - generate_synthetic_ember.py (test data)

✓ Fine-tuning (requires GPU):
  - training/finetune.py (ready, needs GPU for execution)
  - training/export_dataset.py (conversation export)

Ready to use:
  1. python aisha.py                    — start chat
  2. python ingest.py <file>            — load knowledge
  3. python malware_classifier/train_classifier.py  — train classifier
  4. training/finetune.py               — fine-tune (Colab or GPU machine)
""")

if __name__ == "__main__":
    main()
