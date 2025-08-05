#!/usr/bin/env python3
"""
Launcher script for chatDESI application.
Run this file instead of chatdesi/main.py directly.
"""

import sys
import os

# Add the current directory to Python path so we can import chatdesi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run the main app
from chatdesi.main import run_streamlit_app

if __name__ == "__main__":
    run_streamlit_app()