#!/usr/bin/env python3
"""Increment CuratorKIT's patch version for an automated release PR."""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INIT_FILE = ROOT / "curatorkit" / "__init__.py"
VERSION_RE = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def next_patch(version: str) -> str:
    match = SEMVER_RE.fullmatch(version)
    if not match:
        raise SystemExit(f"Current version is not plain SemVer: {version}")

    major, minor, patch = (int(part) for part in match.groups())
    return f"{major}.{minor}.{patch + 1}"


def bump_version(init_file: Path = INIT_FILE) -> str:
    text = init_file.read_text()
    match = VERSION_RE.search(text)
    if not match:
        raise SystemExit(f"Could not find __version__ in {init_file}")

    version = next_patch(match.group(1))
    init_file.write_text(VERSION_RE.sub(f'__version__ = "{version}"', text, count=1))
    return version


def main() -> None:
    version = bump_version()
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as output:
            output.write(f"version={version}\n")
    print(version)


if __name__ == "__main__":
    main()
