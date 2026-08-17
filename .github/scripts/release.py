#!/usr/bin/env python3
"""Tag the version already committed to main and publish a GitHub release."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INIT_FILE = ROOT / "curatorkit" / "__init__.py"
VERSION_RE = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def run(cmd: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def current_version() -> str:
    match = VERSION_RE.search(INIT_FILE.read_text())
    if not match:
        raise SystemExit(f"Could not find __version__ in {INIT_FILE}")
    return match.group(1)


def validate_version(version: str) -> str:
    version = version.removeprefix("v").strip()
    if not SEMVER_RE.fullmatch(version):
        raise SystemExit(
            f"Release version must be plain SemVer without prerelease/build metadata: {version}"
        )
    return version


def ensure_clean_worktree() -> None:
    status = run(["git", "status", "--porcelain"], capture=True)
    if status:
        raise SystemExit("Working tree must be clean before creating a release")


def release_version(requested_version: str | None) -> str:
    package_version = validate_version(current_version())
    if not requested_version:
        return package_version

    requested_version = validate_version(requested_version)
    if requested_version != package_version:
        raise SystemExit(
            f"Requested release version {requested_version} does not match "
            f"package version {package_version}. Bump curatorkit/__init__.py "
            "through a pull request first."
        )

    return requested_version


def ensure_tag_available(version: str) -> None:
    existing = run(["git", "tag", "--list", version], capture=True)
    if existing:
        raise SystemExit(f"Tag already exists locally: {version}")

    remote = run(["git", "ls-remote", "--tags", "origin", version], capture=True)
    if remote:
        raise SystemExit(f"Tag already exists on origin: {version}")


def configure_git_identity() -> None:
    run(["git", "config", "user.name", "github-actions[bot]"])
    run(
        [
            "git",
            "config",
            "user.email",
            "41898282+github-actions[bot]@users.noreply.github.com",
        ]
    )


def create_release(version: str) -> None:
    run(["git", "tag", "-a", version, "-m", f"Release {version}"])
    run(["git", "push", "origin", version])
    run(["gh", "release", "create", version, "--generate-notes", "--latest"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        help="Version to release. Defaults to the version in curatorkit/__init__.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = release_version(args.version)

    ensure_clean_worktree()
    ensure_tag_available(version)
    configure_git_identity()
    create_release(version)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as output:
            output.write(f"version={version}\n")

    print(f"Created release {version}")


if __name__ == "__main__":
    main()
