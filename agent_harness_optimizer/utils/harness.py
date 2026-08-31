"""Harness identity hash — stable fingerprint for (system_prompt, middleware) pair."""

from __future__ import annotations

import hashlib
from pathlib import Path


def harness_hash(system_prompt: str, middleware_dir: Path | None) -> str:
    """Return a 12-char MD5 hex digest of system_prompt + sorted middleware *.py files."""
    h = hashlib.md5(system_prompt.encode())
    if middleware_dir and middleware_dir.is_dir():
        for f in sorted(middleware_dir.glob("*.py")):
            h.update(f.read_bytes())
    return h.hexdigest()[:12]
