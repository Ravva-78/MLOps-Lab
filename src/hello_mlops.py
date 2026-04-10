#!/usr/bin/env python3
"""
Lab 1: MLOps Environment Verification
Confirms that all core libraries are installed and working.
"""

import sys
import pandas as pd
import numpy as np
import sklearn
import dvc.api
import mlflow

print("=" * 60)
print("✓ MLOps Environment Ready!")
print("=" * 60)
print()

print("Python Environment:")
print(f"  Python version:  {sys.version.split()[0]}")
print(f"  Python location: {sys.executable}")
print()

print("Core Libraries:")
print(f"  pandas:          {pd.__version__}")
print(f"  numpy:           {np.__version__}")
print(f"  scikit-learn:    {sklearn.__version__}")
print(f"  dvc:             {dvc.__version__}")
print(f"  mlflow:          {mlflow.__version__}")
print()

print("=" * 60)
print("Ready to start MLOps adventures! 🚀")
print("=" * 60)