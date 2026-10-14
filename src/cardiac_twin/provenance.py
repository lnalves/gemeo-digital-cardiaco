"""Provenance of generated artifacts (CLAUDE.md, rule 7)."""

from __future__ import annotations

import subprocess

from cardiac_twin.config import ROOT


def git_hash() -> str:
    """Current commit, with a `-dirty` suffix if tracked files have uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        head = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}-dirty" if dirty else head
