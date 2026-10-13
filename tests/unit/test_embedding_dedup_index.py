"""
Unit tests for the persisted cross-run index of EmbeddingDeduplicator:
manifest + model/dimension checks, no silent fresh start, atomic generations,
merge-on-save, deferred commit, namespaces, and the Curator wiring
(commit only on success, survive a checkpoint resume).

A deterministic fake embedding model stands in for sentence-transformers.
"""

from __future__ import annotations

import json
import warnings

import pytest

np = pytest.importorskip("numpy")

from curatorkit.normalizers.embedding_dedup import (  # noqa: E402
    EmbeddingDeduplicator,
    EmbeddingIndexError,
    EmbeddingIndexMismatchError,
)
from curatorkit.schema import DataSample  # noqa: E402

try:
    import faiss  # noqa: F401

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class _FakeEmbeddingModel:
    """Bag-of-words hashing 'embedding': identical text -> identical vector."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                vecs[i, sum(map(ord, word)) % self.dim] += 1.0
        return vecs


def make_sample(
    instruction: str, output: str = "a reasonably long answer for testing"
) -> DataSample:
    return DataSample(source_uri="test://", instruction=instruction, output=output)


def samples(*words: str) -> list[DataSample]:
    return [make_sample(f"Explain the topic of {w} in detail please") for w in words]


def dedup(index_dir, dim: int = 64, **kw) -> EmbeddingDeduplicator:
    d = EmbeddingDeduplicator(index_dir=index_dir, threshold=0.99, **kw)
    d._model = _FakeEmbeddingModel(dim)
    return d


def manifest(index_dir) -> dict:
    return json.loads((index_dir / "index_meta.json").read_text())


# ── Basic cross-run behaviour + manifest ─────────────────────────────────────


def test_cross_run_dedup_and_manifest(tmp_path):
    idx = tmp_path / "idx"
    assert len(dedup(idx).run(samples("alpha", "beta"))) == 2
    m = manifest(idx)
    assert m["count"] == 2 and m["dim"] == 64 and m["version"] == 1
    assert m["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert (idx / m["generation"]).is_dir()

    second = dedup(idx).run(samples("alpha", "gamma"))  # alpha is a cross-run duplicate
    assert [s.instruction for s in second] == [samples("gamma")[0].instruction]
    assert manifest(idx)["count"] == 3


def test_old_generations_are_pruned(tmp_path):
    idx = tmp_path / "idx"
    for word in ("a1", "a2", "a3", "a4"):
        dedup(idx, keep_generations=2).run(samples(word))
    gens = sorted(p.name for p in idx.glob("gen-*"))
    assert gens == ["gen-000003", "gen-000004"]
    assert manifest(idx)["count"] == 4


# ── Model / dimension mismatch ───────────────────────────────────────────────


def test_different_model_raises(tmp_path):
    idx = tmp_path / "idx"
    dedup(idx).run(samples("alpha"))
    other = EmbeddingDeduplicator(index_dir=idx, model="some/other-model", threshold=0.99)
    other._model = _FakeEmbeddingModel()
    with pytest.raises(EmbeddingIndexMismatchError, match="some/other-model"):
        other.run(samples("beta"))


def test_different_dimension_raises(tmp_path):
    idx = tmp_path / "idx"
    dedup(idx, dim=64).run(samples("alpha"))
    with pytest.raises(EmbeddingIndexMismatchError, match="64-dimensional"):
        dedup(idx, dim=32).run(samples("beta"))


# ── Unreadable index: never silently start fresh ─────────────────────────────


def test_corrupt_index_raises_and_is_left_untouched(tmp_path):
    idx = tmp_path / "idx"
    dedup(idx).run(samples("alpha", "beta"))
    meta_file = idx / manifest(idx)["generation"] / "metadata.json"
    meta_file.write_text("{not json")

    with pytest.raises(EmbeddingIndexError):
        dedup(idx).run(samples("gamma"))
    assert meta_file.read_text() == "{not json"  # nothing overwritten
    assert manifest(idx)["count"] == 2


def test_missing_generation_raises(tmp_path):
    idx = tmp_path / "idx"
    dedup(idx).run(samples("alpha"))
    import shutil

    shutil.rmtree(idx / manifest(idx)["generation"])
    with pytest.raises(EmbeddingIndexError, match="missing"):
        dedup(idx).run(samples("beta"))


def test_rebuild_moves_unreadable_index_aside(tmp_path):
    idx = tmp_path / "idx"
    dedup(idx).run(samples("alpha"))
    (idx / "index_meta.json").write_text("{broken")

    with pytest.warns(UserWarning, match="moved it to"):
        out = dedup(idx, on_unreadable_index="rebuild").run(samples("alpha"))
    assert len(out) == 1  # fresh index, so not a duplicate
    quarantined = list(idx.glob("unreadable-*"))
    assert len(quarantined) == 1 and (quarantined[0] / "index_meta.json").exists()
    assert manifest(idx)["count"] == 1


# ── Legacy (pre-manifest) flat layout ────────────────────────────────────────


def _write_legacy_index(idx, n: int, dim: int = 64):
    idx.mkdir(parents=True)
    vecs = np.eye(n, dim, dtype=np.float32)
    np.save(str(idx / "embeddings.npy"), vecs)
    (idx / "metadata.json").write_text(
        json.dumps([{"sample_id": f"old-{i}", "source_uri": "x"} for i in range(n)])
    )


@pytest.mark.skipif(_HAS_FAISS, reason="legacy numpy layout is read through the numpy path here")
def test_legacy_index_is_kept_and_migrated(tmp_path):
    idx = tmp_path / "idx"
    _write_legacy_index(idx, n=3)
    with pytest.warns(UserWarning, match="no manifest"):
        dedup(idx).run(samples("alpha"))
    m = manifest(idx)
    assert m["count"] == 4  # 3 legacy entries preserved + 1 new
    assert not (idx / "embeddings.npy").exists() and not (idx / "metadata.json").exists()


def test_legacy_incomplete_index_raises(tmp_path):
    idx = tmp_path / "idx"
    idx.mkdir()
    np.save(str(idx / "embeddings.npy"), np.eye(2, 64, dtype=np.float32))  # no metadata.json
    with pytest.raises(EmbeddingIndexError, match="incomplete"):
        dedup(idx).run(samples("alpha"))


# ── Merge on save (concurrent runs) ──────────────────────────────────────────


def test_concurrent_runs_do_not_lose_entries(tmp_path):
    idx = tmp_path / "idx"
    dedup(idx).run(samples("base"))

    run_a, run_b = dedup(idx), dedup(idx)
    run_a._load_index()  # both load the same 1-entry index...
    run_b._load_index()
    run_a.run(samples("from-a"))
    run_b.run(samples("from-b"))  # ...B's save re-reads and keeps A's entry

    assert manifest(idx)["count"] == 3
    meta = json.loads((idx / manifest(idx)["generation"] / "metadata.json").read_text())
    assert len({m["sample_id"] for m in meta}) == 3


def test_saving_same_entries_twice_is_a_noop(tmp_path):
    idx = tmp_path / "idx"
    d = dedup(idx)
    batch = np.eye(2, 64, dtype=np.float32)
    meta = [{"sample_id": "s1", "source_uri": "x"}, {"sample_id": "s2", "source_uri": "x"}]
    assert d._save_index(batch, meta) == 2
    assert d._save_index(batch, meta) == 0
    assert manifest(idx)["count"] == 2


# ── Deferred commit ──────────────────────────────────────────────────────────


def test_defer_commit_stages_then_commits_once(tmp_path):
    idx = tmp_path / "idx"
    d = dedup(idx, defer_commit=True)
    assert len(d.run(samples("alpha", "beta"))) == 2
    assert not (idx / "index_meta.json").exists()  # nothing committed yet
    assert d.pending_count() == 2

    assert d.commit_pending() == 2
    assert manifest(idx)["count"] == 2
    assert d.pending_count() == 0
    assert d.commit_pending() == 0  # idempotent


def test_defer_commit_dedups_against_earlier_staged_entries(tmp_path):
    # e.g. reward-refiner recoveries, deduped after the pipeline's own dedup step
    d = dedup(tmp_path / "idx", defer_commit=True)
    d.run(samples("alpha"))
    out = d.run(samples("alpha", "gamma"))  # "alpha" is already staged in this run
    assert [s.instruction for s in out] == [samples("gamma")[0].instruction]
    assert d.pending_count() == 2


def test_discard_pending_commits_nothing(tmp_path):
    idx = tmp_path / "idx"
    d = dedup(idx, defer_commit=True)
    d.run(samples("alpha"))
    d.discard_pending()
    assert d.commit_pending() == 0
    assert not (idx / "index_meta.json").exists()


# ── Namespaces, stats, reset ─────────────────────────────────────────────────


def test_namespaces_are_isolated(tmp_path):
    idx = tmp_path / "idx"
    dedup(idx, namespace="contracts-qa").run(samples("alpha"))
    out = dedup(idx, namespace="orca-sft").run(samples("alpha"))
    assert len(out) == 1  # different namespace → not a duplicate
    assert manifest(idx / "contracts-qa")["count"] == 1
    assert manifest(idx / "orca-sft")["count"] == 1


def test_invalid_namespace_raises(tmp_path):
    with pytest.raises(ValueError, match="namespace"):
        EmbeddingDeduplicator(index_dir=tmp_path, namespace="../escape")


def test_stats_and_reset(tmp_path):
    idx = tmp_path / "idx"
    d = dedup(idx)
    d.run(samples("alpha", "beta"))
    s = d.stats()
    assert s["entries"] == 2 and s["dim"] == 64 and s["version"] == 1 and s["pending"] == 0

    d.reset()
    assert d.stats()["entries"] == 0
    assert len(dedup(idx).run(samples("alpha"))) == 1


# ── Curator wiring: commit only on success, survive resume ───────────────────


def _curator(tmp_path, monkeypatch, out_name: str, **kw):
    from curatorkit.curator import Curator, CuratorConfig

    monkeypatch.setattr(EmbeddingDeduplicator, "_load_model", lambda self: _FakeEmbeddingModel())
    data = tmp_path / "data.jsonl"
    if not data.exists():
        rows = [
            {
                "instruction": f"Describe the history of topic number {w} thoroughly",
                "output": f"Topic {w} has a long and interesting history that spans many years.",
            }
            for w in ("one", "two", "three", "four")
        ]
        data.write_text("\n".join(json.dumps(r) for r in rows))
    cfg = CuratorConfig(
        dataset=str(data),
        output_dir=str(tmp_path / out_name),
        embedding_dedup=True,
        embedding_index_dir=str(tmp_path / "index"),
        embedding_dedup_threshold=0.99,
        export_formats=["alpaca"],
        **kw,
    )
    return Curator(cfg)


def test_curator_commits_only_after_success(tmp_path, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _curator(tmp_path, monkeypatch, "run1").run()
    assert len(result.passed) == 4
    assert result.stage_counts["EmbeddingDeduplicator"]["index_committed"] == 4
    assert manifest(tmp_path / "index")["count"] == 4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rerun = _curator(tmp_path, monkeypatch, "run2").run()
    assert len(rerun.passed) == 0  # everything is a cross-run duplicate now


def test_failed_curator_run_adds_nothing(tmp_path, monkeypatch):
    from curatorkit.curator import Curator

    def boom(self, *a, **k):
        raise RuntimeError("export failed")

    monkeypatch.setattr(Curator, "_write_provenance", boom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="export failed"):
            _curator(tmp_path, monkeypatch, "run1").run()
    assert not (tmp_path / "index" / "index_meta.json").exists()

    # a fresh retry must not commit the failed attempt's staged entries — and must
    # not reject its own samples as duplicates of them
    monkeypatch.undo()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        retry = _curator(tmp_path, monkeypatch, "run1").run()
    assert len(retry.passed) == 4
    assert manifest(tmp_path / "index")["count"] == 4


def test_staged_entries_survive_a_checkpoint_resume(tmp_path, monkeypatch):
    from curatorkit.curator import Curator

    calls = {"n": 0}
    real = Curator._write_provenance

    def fail_once(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash after the pipeline")
        return real(self, *a, **k)

    monkeypatch.setattr(Curator, "_write_provenance", fail_once)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            _curator(tmp_path, monkeypatch, "run1", enable_checkpoint=True).run()
        assert not (tmp_path / "index" / "index_meta.json").exists()

        # resume: the dedup stage is restored from the checkpoint (not re-run), yet
        # the entries it staged are still committed once the run succeeds
        resumed = _curator(tmp_path, monkeypatch, "run1", enable_checkpoint=True).run()
    assert len(resumed.passed) == 4
    assert manifest(tmp_path / "index")["count"] == 4


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss-cpu not installed")
def test_faiss_storage_roundtrip(tmp_path):
    idx = tmp_path / "idx"
    dedup(idx).run(samples("alpha", "beta"))
    assert manifest(idx)["storage"] == "faiss"
    assert len(dedup(idx).run(samples("alpha", "gamma"))) == 1


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss-cpu not installed")
def test_legacy_numpy_index_is_read_when_faiss_is_installed(tmp_path):
    idx = tmp_path / "idx"
    _write_legacy_index(idx, n=3)
    with pytest.warns(UserWarning, match="no manifest"):
        dedup(idx).run(samples("alpha"))
    assert manifest(idx)["count"] == 4  # numpy-era entries kept, not silently dropped
