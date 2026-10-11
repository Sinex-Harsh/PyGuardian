"""
__main__.py — Entry point for running PyGuardian via `python -m pyguardian`.
"""

import sys
from pyguardian.cli import main

if __name__ == "__main__":
    sys.exit(main())
