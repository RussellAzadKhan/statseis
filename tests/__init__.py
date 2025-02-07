# tests/__init__.py
import sys
import os

# Ensure the main module is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))