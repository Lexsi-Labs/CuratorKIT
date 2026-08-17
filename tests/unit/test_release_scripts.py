from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / ".github" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bump_version_increments_patch(tmp_path: Path) -> None:
    module = load_script("bump_version")
    init_file = tmp_path / "__init__.py"
    init_file.write_text('"""Package."""\n\n__version__ = "1.2.3"\n')

    assert module.bump_version(init_file) == "1.2.4"
    assert '__version__ = "1.2.4"' in init_file.read_text()


def test_bump_version_rejects_non_semver(tmp_path: Path) -> None:
    module = load_script("bump_version")
    init_file = tmp_path / "__init__.py"
    init_file.write_text('__version__ = "1.2.3-beta"\n')

    with pytest.raises(SystemExit, match="not plain SemVer"):
        module.bump_version(init_file)


def test_release_version_must_match_package_version() -> None:
    module = load_script("release")
    package_version = module.current_version()
    major, minor, patch = (int(part) for part in package_version.split("."))
    different_version = f"{major}.{minor}.{patch + 1}"

    assert module.release_version(None) == package_version
    assert module.release_version(f"v{package_version}") == package_version
    with pytest.raises(SystemExit, match=f"does not match package version {package_version}"):
        module.release_version(different_version)
