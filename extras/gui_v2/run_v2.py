"""
Cardiotect V2 - Application Entry Point
"""

import sys
import os

# Ensure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui_v2.main_window import run_app


if __name__ == "__main__":
    run_app()
