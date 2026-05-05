"""
Agent 29 - Day 1 Smoke Test
Verifies all key dependencies import and basic functionality works.
"""
import sys
from datetime import datetime

print("=" * 60)
print("AGENT 29 SMOKE TEST")
print(f"Python: {sys.version.split()[0]}")
print(f"Time:   {datetime.now().isoformat(timespec='seconds')}")
print("=" * 60)

results = []

def check(name, func):
    try:
        info = func()
        print(f"  [OK]   {name:30s} {info}")
        results.append((name, True, info))
    except Exception as e:
        print(f"  [FAIL] {name:30s} {type(e).__name__}: {e}")
        results.append((name, False, str(e)))


# Core data
check("pandas",        lambda: __import__("pandas").__version__)
check("numpy",         lambda: __import__("numpy").__version__)
check("fastparquet",   lambda: __import__("fastparquet").__version__)
check("pyarrow",       lambda: __import__("pyarrow").__version__)

# ML
check("scikit-learn",  lambda: __import__("sklearn").__version__)
check("lightgbm",      lambda: __import__("lightgbm").__version__)
check("xgboost",       lambda: __import__("xgboost").__version__)
check("joblib",        lambda: __import__("joblib").__version__)

# Deep learning
def torch_info():
    import torch
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "CPU only"
    return f"v{torch.__version__}  CUDA={cuda}  ({device})"
check("torch",         torch_info)
check("transformers",  lambda: __import__("transformers").__version__)

# Brokers
check("alpaca-py",     lambda: __import__("alpaca").__version__)
check("python-binance",lambda: __import__("binance").__version__)

# Market data
check("yfinance",      lambda: __import__("yfinance").__version__)
check("finnhub",       lambda: __import__("finnhub").__version__ if hasattr(__import__("finnhub"), "__version__") else "imported")

# News + sentiment
check("newsapi",       lambda: "imported" if __import__("newsapi") else "fail")
check("praw",          lambda: __import__("praw").__version__)

# Macro
check("fredapi",       lambda: "imported" if __import__("fredapi") else "fail")

# Indicators
check("ta",            lambda: __import__("ta").__version__ if hasattr(__import__("ta"), "__version__") else "imported")

# Telegram
check("telegram",      lambda: __import__("telegram").__version__)

# Dashboard
check("streamlit",     lambda: __import__("streamlit").__version__)
check("plotly",        lambda: __import__("plotly").__version__)

# Config + utils
check("dotenv",        lambda: "imported" if __import__("dotenv") else "fail")
check("yaml",          lambda: __import__("yaml").__version__)
check("loguru",        lambda: "imported" if __import__("loguru") else "fail")
check("apscheduler",   lambda: __import__("apscheduler").__version__)

print("=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"RESULT: {passed} passed, {failed} failed (out of {len(results)})")
print("=" * 60)

if failed > 0:
    print("\nFailures:")
    for name, ok, info in results:
        if not ok:
            print(f"  - {name}: {info}")
    sys.exit(1)
else:
    print("\nAll core libraries are operational. Day 1 environment ready.")
    sys.exit(0)
