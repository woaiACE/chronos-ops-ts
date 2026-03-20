"""Backward-compatible shim.

Prefer running `python main.py`.
This file is kept so existing commands like `python forecast.py ...` still work.
"""

from src.forecast import main


if __name__ == "__main__":
    main()
