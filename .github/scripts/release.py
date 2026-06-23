#!/usr/bin/env python3
"""Create a version bump commit, tag it, and publish a GitHub release."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INIT_FILE = ROOT / "curatorkit" / "__init__.py"
INITIAL_VERSION = "0.1.0"
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


def next_patch(version: str) -> str:
    match = SEMVER_RE.fullmatch(version)
    if not match:
        raise SystemExit(f"Current version is not plain SemVer: {version}")

    major, minor, patch = (int(part) for part in match.groups())
    return f"{major}.{minor}.{patch + 1}"


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


def semver_tags() -> list[str]:
    output = run(["git", "tag", "--list", "*.*.*", "--sort=-v:refname"], capture=True)
    return [tag for tag in output.splitlines() if SEMVER_RE.fullmatch(tag)]


def release_version(requested_version: str | None) -> str:
    if requested_version:
        return validate_version(requested_version)

    tags = semver_tags()
    if tags:
        return next_patch(tags[0])

    return INITIAL_VERSION


def ensure_tag_available(version: str) -> None:
    existing = run(["git", "tag", "--list", version], capture=True)
    if existing:
        raise SystemExit(f"Tag already exists locally: {version}")

    remote = run(["git", "ls-remote", "--tags", "origin", version], capture=True)
    if remote:
        raise SystemExit(f"Tag already exists on origin: {version}")


def write_version(version: str) -> bool:
    text = INIT_FILE.read_text()
    updated = VERSION_RE.sub(f'__version__ = "{version}"', text, count=1)
    if text == updated:
        return False
    INIT_FILE.write_text(updated)
    return True


def configure_git_identity() -> None:
    run(["git", "config", "user.name", "github-actions[bot]"])
    run(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"])


def create_release(version: str) -> None:
    changed = write_version(version)

    if changed:
        run(["git", "add", str(INIT_FILE.relative_to(ROOT))])
        run(["git", "commit", "-m", f"Release {version} [skip ci]"])
        run(["git", "push", "origin", "HEAD:main"])

    run(["git", "tag", "-a", version, "-m", f"Release {version}"])
    run(["git", "push", "origin", version])
    run(["gh", "release", "create", version, "--generate-notes", "--latest"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        help="Version to release, without a leading v. Defaults to the next patch version.",
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
