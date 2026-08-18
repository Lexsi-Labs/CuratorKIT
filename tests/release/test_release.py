import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.release import (
    Presence,
    ReleaseAction,
    ReleaseError,
    ReleaseState,
    classify_release_state,
    current_version,
    decide_action,
    next_patch,
    parse_version,
    set_version,
    validate_distributions,
    validate_requested_version,
)


def _version_tree(root: Path, version: str = "1.0.0") -> Path:
    (root / "curatorkit").mkdir(parents=True)
    (root / "curatorkit/__init__.py").write_text(f'__version__ = "{version}"\n')
    (root / "CITATION.cff").write_text(f'version: "{version}"\n')
    return root


@pytest.mark.parametrize(
    ("current", "expected"),
    [("0.1.0", "0.1.1"), ("1.9.99", "1.9.100"), ("10.0.8", "10.0.9")],
)
def test_next_patch_increments_only_patch(current, expected):
    assert next_patch(current) == expected


def test_successive_merges_advance_patch_releases():
    latest = "0.1.0"
    for expected in ("0.1.1", "0.1.2", "0.1.3"):
        requested = next_patch(latest)
        assert requested == expected
        decision = decide_action(trigger="normal", version=requested, state=ReleaseState.READY)
        assert decision.action is ReleaseAction.RELEASE
        assert decision.version == expected
        latest = requested


@pytest.mark.parametrize(
    "invalid",
    ["v1.2.3", "1.2", "1.2.3.4", "01.2.3", "1.02.3", "1.2.03", "1.2.3rc1", "1.2.3+meta", ""],
)
def test_plain_semver_rejects_invalid_versions(invalid):
    with pytest.raises(ReleaseError):
        parse_version(invalid)


def test_initial_version_is_preserved_for_bootstrap(tmp_path):
    root = _version_tree(tmp_path, "0.1.0")
    state = classify_release_state(Presence(False, False, False))
    decision = decide_action(trigger="dispatch", version=current_version(root), state=state)
    assert decision.action is ReleaseAction.RELEASE
    assert decision.version == "0.1.0"
    assert decision.state is ReleaseState.READY


def test_set_version_updates_every_authoritative_location(tmp_path):
    root = _version_tree(tmp_path)
    set_version("1.0.1", root)
    assert current_version(root) == "1.0.1"
    assert "1.0.1" in (root / "curatorkit/__init__.py").read_text()
    assert "1.0.1" in (root / "CITATION.cff").read_text()


def test_set_version_repairs_partially_bumped_metadata(tmp_path):
    root = _version_tree(tmp_path, "0.1.0")
    (root / "curatorkit/__init__.py").write_text('__version__ = "0.1.1"\n')

    with pytest.raises(ReleaseError, match="disagree"):
        current_version(root)

    set_version("0.1.1", root)
    assert current_version(root) == "0.1.1"


def test_requested_version_must_match_all_package_metadata(tmp_path):
    root = _version_tree(tmp_path)
    with pytest.raises(ReleaseError, match="does not match"):
        validate_requested_version("1.0.1", root)
    (root / "CITATION.cff").write_text("version: 1.0.1\n")
    with pytest.raises(ReleaseError, match="disagree"):
        current_version(root)


@pytest.mark.parametrize(
    ("presence", "expected"),
    [
        (Presence(False, False, False), ReleaseState.READY),
        (Presence(True, False, False), ReleaseState.RESUMABLE),
        (Presence(True, True, False), ReleaseState.RESUMABLE),
        (Presence(True, True, True), ReleaseState.COMPLETE),
        (Presence(False, True, False), ReleaseState.CONFLICT),
        (Presence(False, False, True), ReleaseState.CONFLICT),
        (Presence(True, False, True), ReleaseState.CONFLICT),
    ],
)
def test_release_completeness_states(presence, expected):
    assert classify_release_state(presence) is expected


def test_mismatched_or_lightweight_tag_is_conflicting():
    presence = Presence(True, False, False)
    assert classify_release_state(presence, tag_is_annotated=False) is ReleaseState.CONFLICT
    assert classify_release_state(presence, tag_matches_source=False) is ReleaseState.CONFLICT


def test_normal_merge_bootstraps_incomplete_current_version():
    decision = decide_action(trigger="normal", version="1.0.0", state=ReleaseState.READY)
    assert decision.action is ReleaseAction.RELEASE
    assert decision.version == "1.0.0"


def test_normal_merge_resumes_partial_current_release():
    decision = decide_action(trigger="normal", version="1.0.0", state=ReleaseState.RESUMABLE)
    assert decision.action is ReleaseAction.RELEASE
    assert decision.version == "1.0.0"


def test_complete_current_release_is_a_noop():
    decision = decide_action(trigger="normal", version="1.0.0", state=ReleaseState.COMPLETE)
    assert decision.action is ReleaseAction.NOOP
    assert decision.version == "1.0.0"


def test_conflicting_state_fails_decision():
    with pytest.raises(ReleaseError, match="manual repair"):
        decide_action(trigger="dispatch", version="1.0.0", state=ReleaseState.CONFLICT)


def _write_distributions(root: Path, version: str) -> list[Path]:
    metadata = f"Metadata-Version: 2.1\nName: curatorkit\nVersion: {version}\n\n"
    wheel = root / f"curatorkit-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"curatorkit-{version}.dist-info/METADATA", metadata)
    sdist = root / f"curatorkit-{version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        payload = metadata.encode()
        info = tarfile.TarInfo(f"curatorkit-{version}/PKG-INFO")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        nested = tarfile.TarInfo(f"curatorkit-{version}/curatorkit.egg-info/PKG-INFO")
        nested.size = len(payload)
        archive.addfile(nested, io.BytesIO(payload))
    return [wheel, sdist]


def test_distribution_metadata_must_match_requested_version(tmp_path):
    validate_distributions("1.0.0", _write_distributions(tmp_path, "1.0.0"))
    with pytest.raises(ReleaseError, match="expected exact version"):
        validate_distributions("1.0.1", list(tmp_path.iterdir()))
