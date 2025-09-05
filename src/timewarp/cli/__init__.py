from __future__ import annotations

# Expose only the CLI entrypoint. Helper utilities live under cli.helpers.
from .main import main

__all__ = ["main"]
