"""SecretsGate's install hint, tested with detect-secrets made unimportable."""

from __future__ import annotations

import sys

import pytest


def test_missing_detect_secrets_names_the_single_package(monkeypatch):
    from curatorkit.hygiene.secrets import _ensure_detect_secrets

    monkeypatch.setitem(sys.modules, "detect_secrets", None)  # import now raises ImportError
    with pytest.raises(ImportError) as exc:
        _ensure_detect_secrets()
    assert "pip install detect-secrets" in str(exc.value)
    assert "curatorkit[hygiene]" in str(exc.value)
