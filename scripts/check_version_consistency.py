#!/usr/bin/env python3
"""Fail CI if the declared version is not the truth, or the CHANGELOG lies about it.

`scripts/release.py` already refuses to release when the two authoritative files
disagree -- but only at release time. This runs on every PR, so drift is caught
while it is one commit old instead of nine releases old. That is not hypothetical:
the tree sat at 0.1.6 while PyPI served 0.1.15, so every wheel since 0.1.7
reported a `curatorkit.__version__` that misidentified the release.

Three checks:

1. the two authoritative files declare the same version,
2. that version is valid semver,
3. no dated CHANGELOG heading claims a version higher than the declared one.

Check 3 exists because of `## 1.0.0 - 2026-06-12`: an unshipped version carrying
a past date reads as shipped, which is why CITATION.cff sat at 1.0.0 and looked
consistent with the CHANGELOG rather than wrong.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.release import VERSION_FILES, ReleaseError, read_versions  # noqa: E402

SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
# "## 1.0.0 - 2026-06-12" and "## 0.2.0" both count; "## Unreleased" and any
# section with no leading version number are ignored.
HEADING = re.compile(r"^##\s+(\d+\.\d+\.\d*)\b", re.M)


def parse(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def main() -> int:
    problems: list[str] = []

    # 1 + 2: the authoritative files agree, and say something valid.
    try:
        versions = read_versions(ROOT)
    except ReleaseError as exc:
        print(f"FAIL: {exc}")
        return 1

    distinct = set(versions.values())
    if len(distinct) != 1:
        for path, value in versions.items():
            print(f"  {path} = {value}")
        print(
            "FAIL: the authoritative version files disagree. "
            "`pyproject.toml` derives the package version from "
            f"{VERSION_FILES[0]}, so that file is what pip installs -- "
            "set them both with `python scripts/release.py set-version <v>`."
        )
        return 1

    declared = distinct.pop()
    if not SEMVER.match(declared):
        print(f"FAIL: {declared!r} is not a valid semver version.")
        return 1

    # 3: no dated heading may claim more than what we declare.
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for heading in HEADING.finditer(changelog):
        claimed = heading.group(1)
        try:
            if parse(claimed) > parse(declared):
                problems.append(
                    f"CHANGELOG.md claims {claimed}, but the package declares {declared}."
                    "\n  A dated heading for a version that was never published reads as"
                    "\n  shipped. Move it under '## Unreleased' or drop the date."
                )
        except ValueError:  # a partial version like 0.2 -- not comparable
            continue

    if problems:
        print(f"FAIL: declared version {declared}, but:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"OK: authoritative files agree on {declared}, and no heading overclaims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
