#!/usr/bin/env python3
"""
Shim to keep local `python timewarp_interactive_debug.py` usage working.
The real implementation lives in `timewarp.interactive_debug`.
"""

from __future__ import annotations

from timewarp.interactive_debug import launch_debugger, main

__all__ = ["launch_debugger", "main"]

if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(main())
