"""Pure, testable helpers for CuratorKIT patch releases.

This module deliberately keeps network access and repository mutations out of
the decision functions.  The GitHub Actions workflow gathers remote state and
passes booleans to these helpers, which makes the safety rules easy to test.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


SEMVER_PATTERN = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")

VERSION_FILES = (
    Path("curatorkit/__init__.py"),
    Path("CITATION.cff"),
)


class ReleaseError(ValueError):
    """Raised when version or release state is unsafe or inconsistent."""


class ReleaseState(str, Enum):
    READY = "ready"
    RESUMABLE = "resumable"
    COMPLETE = "complete"
    CONFLICT = "conflict"


class ReleaseAction(str, Enum):
    RELEASE = "release"
    NOOP = "noop"


@dataclass(frozen=True)
class Presence:
    """Whether an exact version exists in each external release system."""

    tag: bool
    github_release: bool
    pypi: bool


@dataclass(frozen=True)
class Decision:
    action: ReleaseAction
    version: str
    state: ReleaseState
    reason: str


def parse_version(value: str) -> tuple[int, int, int]:
    """Parse a plain MAJOR.MINOR.PATCH version, rejecting all extensions."""

    match = SEMVER_PATTERN.fullmatch(value)
    if not match:
        raise ReleaseError(
            f"Invalid version {value!r}; expected plain SemVer MAJOR.MINOR.PATCH "
            "with no prefix, suffix, build metadata, or leading zeroes."
        )
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def next_patch(value: str) -> str:
    major, minor, patch = parse_version(value)
    return f"{major}.{minor}.{patch + 1}"


def _single_match(pattern: re.Pattern[str], text: str, location: Path) -> str:
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ReleaseError(
            f"Expected exactly one authoritative version in {location}, found {len(matches)}."
        )
    value = matches[0]
    if isinstance(value, tuple):
        value = ".".join(value)
    parse_version(value)
    return value


_INIT_VERSION = re.compile(r'(?m)^__version__\s*=\s*"([^"]+)"\s*$')
_CITATION_VERSION = re.compile(r'''(?m)^version:\s*["']?([^"'\s#]+)["']?\s*$''')


def read_versions(root: Path | str = Path(".")) -> dict[Path, str]:
    root = Path(root)
    patterns = {
        VERSION_FILES[0]: _INIT_VERSION,
        VERSION_FILES[1]: _CITATION_VERSION,
    }
    versions: dict[Path, str] = {}
    for relative, pattern in patterns.items():
        path = root / relative
        if not path.is_file():
            raise ReleaseError(f"Authoritative version file is missing: {relative}")
        versions[relative] = _single_match(pattern, path.read_text(encoding="utf-8"), relative)
    return versions


def current_version(root: Path | str = Path(".")) -> str:
    versions = read_versions(root)
    unique = set(versions.values())
    if len(unique) != 1:
        details = ", ".join(f"{path}={version}" for path, version in versions.items())
        raise ReleaseError(f"Authoritative versions disagree: {details}")
    return unique.pop()


def validate_requested_version(requested: str, root: Path | str = Path(".")) -> str:
    parse_version(requested)
    actual = current_version(root)
    if requested != actual:
        raise ReleaseError(
            f"Requested/tag version {requested} does not match package metadata version {actual}."
        )
    return actual


def set_version(requested: str, root: Path | str = Path(".")) -> None:
    """Atomically prepare and then update all authoritative tracked locations."""

    parse_version(requested)
    root = Path(root)
    read_versions(root)  # Validate every source independently, while allowing normalization.
    patterns = {
        VERSION_FILES[0]: _INIT_VERSION,
        VERSION_FILES[1]: _CITATION_VERSION,
    }
    prepared: dict[Path, str] = {}
    for relative, pattern in patterns.items():
        path = root / relative
        text = path.read_text(encoding="utf-8")
        replacement = (
            f'__version__ = "{requested}"'
            if relative == VERSION_FILES[0]
            else f"version: {requested}"
        )
        updated, count = pattern.subn(replacement, text)
        if count != 1:
            raise ReleaseError(f"Could not safely update the version in {relative}.")
        prepared[path] = updated
    for path, text in prepared.items():
        path.write_text(text, encoding="utf-8")
    validate_requested_version(requested, root)


def classify_release_state(
    presence: Presence,
    *,
    tag_is_annotated: bool = True,
    tag_matches_source: bool = True,
    release_matches_tag: bool = True,
) -> ReleaseState:
    """Classify exact-version state without making an external mutation."""

    if presence.tag and (not tag_is_annotated or not tag_matches_source):
        return ReleaseState.CONFLICT
    if presence.github_release and (not presence.tag or not release_matches_tag):
        return ReleaseState.CONFLICT
    if presence.pypi and not (presence.tag and presence.github_release):
        return ReleaseState.CONFLICT
    if presence.tag and presence.github_release and presence.pypi:
        return ReleaseState.COMPLETE
    if presence.tag or presence.github_release:
        return ReleaseState.RESUMABLE
    if presence.pypi:
        return ReleaseState.CONFLICT
    return ReleaseState.READY


def decide_action(
    *,
    trigger: str,
    version: str,
    state: ReleaseState,
) -> Decision:
    """Release the declared version once; completed versions are a no-op."""

    parse_version(version)
    if state is ReleaseState.CONFLICT:
        raise ReleaseError(
            f"Release {version} has conflicting tag/GitHub/PyPI state; manual repair is required."
        )
    if trigger not in {"normal", "dispatch"}:
        raise ReleaseError(f"Unknown release trigger: {trigger!r}")
    if state is ReleaseState.COMPLETE:
        return Decision(ReleaseAction.NOOP, version, state, "current release already complete")
    return Decision(ReleaseAction.RELEASE, version, state, "create or resume declared version")


def _metadata_version(contents: str, filename: Path) -> tuple[str, str]:
    metadata = Parser().parsestr(contents)
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ReleaseError(f"Distribution metadata in {filename} lacks Name or Version.")
    return name, version


def distribution_metadata(path: Path) -> tuple[str, str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if len(candidates) != 1:
                raise ReleaseError(
                    f"Expected one METADATA file in {path}, found {len(candidates)}."
                )
            contents = archive.read(candidates[0]).decode("utf-8")
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            candidates = [
                member
                for member in archive.getmembers()
                if member.name.endswith("/PKG-INFO") and member.name.count("/") == 1
            ]
            if len(candidates) != 1:
                raise ReleaseError(
                    f"Expected one PKG-INFO file in {path}, found {len(candidates)}."
                )
            extracted = archive.extractfile(candidates[0])
            if extracted is None:
                raise ReleaseError(f"Could not read PKG-INFO from {path}.")
            contents = extracted.read().decode("utf-8")
    else:
        raise ReleaseError(f"Unsupported distribution file: {path}")
    return _metadata_version(contents, path)


def validate_distributions(
    requested: str,
    paths: Iterable[Path],
    *,
    project_name: str = "curatorkit",
) -> None:
    parse_version(requested)
    distributions = list(paths)
    if not distributions:
        raise ReleaseError("No distributions were supplied for validation.")

    def normalize(value: str) -> str:
        return re.sub(r"[-_.]+", "-", value).lower()

    for path in distributions:
        name, version = distribution_metadata(path)
        if normalize(name) != normalize(project_name):
            raise ReleaseError(
                f"Distribution {path} has project name {name!r}, expected {project_name!r}."
            )
        if version != requested:
            raise ReleaseError(
                f"Distribution {path} has version {version!r}, expected exact version {requested!r}."
            )


def _bool(value: str) -> bool:
    normalized = value.lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return normalized == "true"


def _json_decision(decision: Decision) -> str:
    return json.dumps(
        {
            "action": decision.action.value,
            "version": decision.version,
            "state": decision.state.value,
            "reason": decision.reason,
        },
        sort_keys=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("current-version")
    subparsers.add_parser("next-version")

    set_parser = subparsers.add_parser("set-version")
    set_parser.add_argument("version")
    validate_parser = subparsers.add_parser("validate-version")
    validate_parser.add_argument("version")
    dist_parser = subparsers.add_parser("validate-dist")
    dist_parser.add_argument("version")
    dist_parser.add_argument("paths", nargs="+", type=Path)
    dist_parser.add_argument("--project-name", default="curatorkit")

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--trigger", choices=("normal", "dispatch"), required=True)
    plan_parser.add_argument("--version", required=True)
    plan_parser.add_argument("--tag", type=_bool, required=True)
    plan_parser.add_argument("--release", type=_bool, required=True)
    plan_parser.add_argument("--pypi", type=_bool, required=True)
    plan_parser.add_argument("--tag-annotated", type=_bool, default=True)
    plan_parser.add_argument("--tag-matches-source", type=_bool, default=True)
    plan_parser.add_argument("--release-matches-tag", type=_bool, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "current-version":
        print(current_version())
    elif args.command == "next-version":
        print(next_patch(current_version()))
    elif args.command == "set-version":
        set_version(args.version)
    elif args.command == "validate-version":
        print(validate_requested_version(args.version))
    elif args.command == "validate-dist":
        validate_requested_version(args.version)
        validate_distributions(args.version, args.paths, project_name=args.project_name)
    elif args.command == "plan":
        state = classify_release_state(
            Presence(args.tag, args.release, args.pypi),
            tag_is_annotated=args.tag_annotated,
            tag_matches_source=args.tag_matches_source,
            release_matches_tag=args.release_matches_tag,
        )
        decision = decide_action(
            trigger=args.trigger,
            version=args.version,
            state=state,
        )
        print(_json_decision(decision))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as error:
        raise SystemExit(f"release error: {error}") from error
