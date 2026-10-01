"""CuratorResult.push_to_hub — real export + load, mocked Hub upload (no network)."""

import json

import pytest

datasets = pytest.importorskip("datasets")

from curatorkit.curator import Curator, CuratorConfig  # noqa: E402


def _seed(path, n=10):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "instruction": f"Explain topic number {i} in a few clear and simple sentences please.",
                "output": f"Topic {i} is explained here in several plain words for the reader to follow.",
            }) + "\n")


@pytest.mark.parametrize("split", [None, {"train": 0.8, "val": 0.2}])
def test_push_to_hub(tmp_path, monkeypatch, split):
    seed = tmp_path / "seed.jsonl"
    _seed(seed)
    result = Curator(CuratorConfig(
        dataset=str(seed), output_dir=str(tmp_path / "out"), output_split=split,
    )).run()
    assert result.passed

    calls = []

    def fake_push(self, repo_id, **kw):
        calls.append((self, repo_id, kw))
        return "ok"

    monkeypatch.setattr(datasets.DatasetDict, "push_to_hub", fake_push)
    assert result.push_to_hub("org/my-data", token="dummy-token") == "ok"

    (ds, repo_id, kw) = calls[0]
    assert repo_id == "org/my-data"
    assert kw["private"] is True and kw["token"] == "dummy-token"
    assert kw["config_name"] == "sft_alpaca"
    assert set(ds) == ({"train", "val"} if split else {"train"})
    assert sum(len(d) for d in ds.values()) == len(result.passed)
    assert set(ds["train"].column_names) == {"instruction", "input", "output"}


def test_push_to_hub_missing_file(tmp_path):
    from curatorkit.curator import CuratorResult
    r = CuratorResult(passed=[], rejected=[], stage_counts={}, output_dir=tmp_path,
                      wall_clock_seconds=0.0)
    with pytest.raises(FileNotFoundError, match="dpo.jsonl"):
        r.push_to_hub("org/x", export_file="dpo.jsonl")
