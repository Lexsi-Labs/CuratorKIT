"""Tests for group-aware splits and the leakage check."""

from __future__ import annotations

import hashlib
import json
import math
import random
import warnings
from collections import Counter

import pytest
from typer.testing import CliRunner

from curatorkit import Curator, CuratorConfig
from curatorkit.schema import DataSample
from curatorkit.splits import leakage_report, split_samples, warn_on_leakage

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _sample(text, group=None, **fields):
    metadata = {} if group is None else {"group": group}
    return DataSample(
        source_uri="test", instruction=text, output="An answer.", metadata=metadata, **fields
    )


def _translated(groups=300, languages=("en", "hi", "bn")):
    return [
        _sample(f"item {g} in {lang}", group=f"g{g}") for g in range(groups) for lang in languages
    ]


def _membership(splits):
    return {s.instruction: name for name, rows in splits.items() for s in rows}


class TestShuffleSplit:
    def test_matches_the_previous_seeded_shuffle(self):
        samples = [_sample(f"item {i}") for i in range(23)]
        shuffled = list(samples)
        random.Random(7).shuffle(shuffled)
        n = len(shuffled)
        train, val = math.floor(n * 0.8), math.floor(n * 0.1)
        splits = split_samples(samples, FRACTIONS, seed=7)
        assert splits["train"] == shuffled[:train]
        assert splits["val"] == shuffled[train : train + val]
        assert splits["test"] == shuffled[train + val :]

    @pytest.mark.parametrize("group_key", [None, "group"])
    def test_fractions_must_sum_to_one(self, group_key):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            split_samples([_sample("x", "g")], {"train": 0.8, "val": 0.1}, group_key=group_key)


class TestGroupSplit:
    def test_every_sample_of_a_group_lands_in_one_split(self):
        splits = split_samples(_translated(), FRACTIONS, group_key="group")
        seen: dict[str, set] = {}
        for name, rows in splits.items():
            for sample in rows:
                seen.setdefault(sample.metadata["group"], set()).add(name)
        assert len(seen) == 300
        assert all(len(names) == 1 for names in seen.values())

    def test_a_per_sample_shuffle_would_have_leaked_here(self):
        splits = split_samples(_translated(), FRACTIONS)
        assert leakage_report(splits, group_key="group", near_dup_threshold=None)[
            "groups_in_multiple_splits"
        ]

    def test_sizes_approximate_the_fractions(self):
        sizes = Counter(
            {
                k: len(v)
                for k, v in split_samples(
                    _translated(2000, ("en",)), FRACTIONS, group_key="group"
                ).items()
            }
        )
        assert sizes["train"] == pytest.approx(1600, abs=80)
        assert sizes["val"] == pytest.approx(200, abs=60)
        assert sum(sizes.values()) == 2000

    def test_the_same_seed_gives_the_same_split_and_another_seed_does_not(self):
        first = _membership(split_samples(_translated(), FRACTIONS, group_key="group", seed=1))
        again = _membership(split_samples(_translated(), FRACTIONS, group_key="group", seed=1))
        other = _membership(split_samples(_translated(), FRACTIONS, group_key="group", seed=2))
        assert first == again
        assert first != other

    def test_an_item_keeps_its_split_when_the_corpus_grows(self):
        small = _membership(split_samples(_translated(100), FRACTIONS, group_key="group"))
        large = _membership(split_samples(_translated(400), FRACTIONS, group_key="group"))
        assert all(large[text] == name for text, name in small.items())

    def test_input_order_is_kept_within_a_split(self):
        samples = _translated(50)
        splits = split_samples(samples, FRACTIONS, group_key="group")
        order = {id(s): i for i, s in enumerate(samples)}
        for rows in splits.values():
            assert [order[id(s)] for s in rows] == sorted(order[id(s)] for s in rows)

    @pytest.mark.filterwarnings("ignore:output_split_group")
    def test_group_values_need_not_be_strings(self):
        samples = [_sample("a", 7), _sample("b", 7), _sample("c", 0)]
        splits = split_samples(samples, {"x": 0.5, "y": 0.5}, group_key="group")
        assert (
            len({name for name, rows in splits.items() for s in rows if s.metadata["group"] == 7})
            == 1
        )
        assert sum(len(rows) for rows in splits.values()) == 3

    def test_a_key_no_sample_has_is_an_error(self):
        with pytest.raises(ValueError, match="no sample has metadata\\['grp'\\]"):
            split_samples(_translated(5), FRACTIONS, group_key="grp")

    def test_a_sample_without_a_group_warns_and_is_split_on_its_text(self):
        samples = [_sample("a", "g1"), _sample("b"), _sample("b"), _sample("c", "")]
        with pytest.warns(UserWarning, match="3 of 4 samples have no"):
            first = split_samples(samples, {"x": 0.5, "y": 0.5}, group_key="group")
        names = [name for name, rows in first.items() for s in rows if s.instruction == "b"]
        assert len(set(names)) == 1  # identical text stays together, though the ids differ

    def test_too_few_groups_for_the_fractions_warns(self):
        with pytest.warns(UserWarning, match="are empty"):
            split_samples([_sample("a", "only")], FRACTIONS, group_key="group")

    def test_no_samples_is_not_an_error(self):
        assert split_samples([], FRACTIONS, group_key="group") == {
            "train": [],
            "val": [],
            "test": [],
        }


class TestLeakageReport:
    def test_clean_splits(self):
        splits = {
            "train": [_sample("first prompt here", "a")],
            "test": [_sample("another one", "b")],
        }
        report = leakage_report(splits, group_key="group")
        assert report == {
            "groups_in_multiple_splits": [],
            "exact_duplicates": [],
            "near_duplicates": [],
            "ok": True,
        }

    def test_a_group_on_both_sides(self):
        splits = {"train": [_sample("in English", "g")], "test": [_sample("auf Deutsch", "g")]}
        report = leakage_report(splits, group_key="group")
        assert report["groups_in_multiple_splits"] == ["g"]
        assert report["ok"] is False

    def test_text_alone_can_not_see_a_translation(self):
        splits = {"train": [_sample("in English", "g")], "test": [_sample("auf Deutsch", "g")]}
        assert leakage_report(splits)["ok"] is True

    def test_exact_duplicates_ignore_case_and_spacing(self):
        a, b = _sample("Tell me  about Rain"), _sample("tell me about rain")
        report = leakage_report({"train": [a], "test": [b]}, near_dup_threshold=None)
        assert report["exact_duplicates"] == [(a.id, b.id)]

    def test_one_prompt_with_two_answers_still_leaks(self):
        a = DataSample(source_uri="t", instruction="same prompt", output="one answer")
        b = DataSample(source_uri="t", instruction="same prompt", output="a different answer")
        assert leakage_report({"train": [a], "test": [b]})["exact_duplicates"] == [(a.id, b.id)]

    def test_a_sample_with_no_prompt_is_compared_on_its_text(self):
        a = DataSample(source_uri="t", task_type="language_modeling", output="the same long text")
        b = DataSample(source_uri="t", task_type="language_modeling", output="the same long text")
        assert leakage_report({"train": [a], "test": [b]})["exact_duplicates"] == [(a.id, b.id)]

    def test_a_duplicate_inside_one_split_is_not_a_leak(self):
        assert (
            leakage_report({"train": [_sample("same"), _sample("same")], "test": []})["ok"] is True
        )

    def test_near_duplicates_are_found_with_their_similarity(self):
        text = "Explain how a bicycle stays upright while it is moving down a long straight road"
        a, b = _sample(text), _sample(text + " today")
        report = leakage_report({"train": [a], "test": [b]})
        assert report["exact_duplicates"] == []
        ((first, second, similarity),) = report["near_duplicates"]
        assert (first, second) == (a.id, b.id)
        assert similarity >= 0.8

    def test_unrelated_text_and_tiny_prompts_are_not_near_duplicates(self):
        splits = {
            "train": [_sample("How do tides work near the coast of Norway"), _sample("a")],
            "test": [_sample("Write a poem about a quiet library in autumn"), _sample("b")],
        }
        assert leakage_report(splits)["near_duplicates"] == []

    def test_a_templated_corpus_stays_linear(self):
        """All pairs here would be about 40,000, and quadratic work on a large corpus."""
        rows = [
            _sample(f"Explain topic number {i} in a few clear and simple sentences please")
            for i in range(400)
        ]
        report = leakage_report({"train": rows[:200], "test": rows[200:]}, num_perm=16)
        assert 0 < len(report["near_duplicates"]) < 400 * 16
        assert report["ok"] is False

    def test_the_near_duplicate_pass_can_be_skipped(self):
        text = "Explain how a bicycle stays upright while it is moving down a long straight road"
        splits = {"train": [_sample(text)], "test": [_sample(text + " today")]}
        assert leakage_report(splits, near_dup_threshold=None)["ok"] is True

    def test_warn_on_leakage(self):
        with pytest.warns(UserWarning, match="1 exact"):
            report = warn_on_leakage({"train": [_sample("same")], "test": [_sample("same")]})
        assert report["ok"] is False

    def test_conversations_are_compared_on_every_user_turn(self):
        def chat(later_user_turn, reply="Sure.", system="You are a helpful assistant."):
            return DataSample(
                source_uri="t",
                task_type="conversational",
                instruction="Hi!",
                output="Hello, how can I help?",
                metadata={"system_prompt": system, "turns": [later_user_turn, reply]},
            )

        user = {"role": "user", "content": "How do tides work near the coast of Norway?"}
        other = {"from": "human", "value": "Write a poem about a quiet library in autumn."}
        reply = {"role": "assistant", "content": "Sure."}
        # Same opener and system prompt, different questions after it: not a leak.
        assert leakage_report({"train": [chat(user, reply)], "test": [chat(other, reply)]})["ok"]
        # Same user turns, different replies and system prompts: the prompt leaks.
        a = chat(user, reply)
        b = chat(user, {"from": "gpt", "value": "No."}, system="Be brief.")
        assert leakage_report({"train": [a], "test": [b]})["exact_duplicates"] == [(a.id, b.id)]

    def test_lightly_edited_prompts_are_found(self):
        """One 9-character word changed in a ~140-character prompt: Jaccard about 0.85."""
        from curatorkit.normalizers.dedup import _ngrams
        from curatorkit.splits import _prompt_text

        train, test = [], []
        for i in range(40):
            words = _words(("near", i)).split()
            train.append(_sample(" ".join(words)))
            words[7] = _words(("edit", i), 1)
            test.append(_sample(" ".join(words)))
        for a, b in zip(train, test):
            x, y = _ngrams(_prompt_text(a)), _ngrams(_prompt_text(b))
            assert len(x & y) / len(x | y) >= 0.82
        found = {b for _, b, _ in leakage_report({"train": train, "test": test})["near_duplicates"]}
        assert len(found) >= 36  # MinHash is approximate; 90% of 40


def _words(key, n=14):
    """Text unlike any other key's, so a clean split has no near duplicates to report."""
    return " ".join(hashlib.sha256(f"{key}/{i}".encode()).hexdigest()[:9] for i in range(n))


def _seed(path, groups=40):
    with open(path, "w", encoding="utf-8") as f:
        for g in range(groups):
            for lang in ("en", "hi"):
                f.write(
                    json.dumps(
                        {
                            "instruction": f"number {g} {_words((g, lang))}",
                            "output": _words((g, lang, "output")),
                            "group": f"g{g}",
                        }
                    )
                    + "\n"
                )


def _groups_per_split(out, names):
    seen: dict[str, set] = {}
    for name in names:
        for line in (out / name / "sft_alpaca.jsonl").read_text(encoding="utf-8").splitlines():
            number = json.loads(line)["instruction"].split("number ")[1].split(" ")[0]
            seen.setdefault(number, set()).add(name)
    return seen


class TestPipeline:
    def test_curator_keeps_a_group_together_from_an_input_column(self, tmp_path):
        seed = tmp_path / "seed.jsonl"
        _seed(seed)
        out = tmp_path / "out"
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # no leak warning on a clean split
            Curator(
                CuratorConfig(
                    dataset=str(seed),
                    output_dir=str(out),
                    export_formats=["alpaca"],
                    dedup=False,
                    output_split={"train": 0.75, "test": 0.25},
                    output_split_group="group",
                )
            ).run()
        seen = _groups_per_split(out, ["train", "test"])
        assert len(seen) == 40
        assert all(len(names) == 1 for names in seen.values())
        provenance = json.loads((out / "lexsi_provenance.json").read_text(encoding="utf-8"))
        assert provenance["params"]["output_split_group"] == "group"

    def test_yaml_runs_take_the_same_path(self, tmp_path):
        from curatorkit.cli import app

        seed = tmp_path / "seed.jsonl"
        _seed(seed)
        (tmp_path / "p.yaml").write_text(
            f"readers:\n- type: jsonl\n  path: {seed.as_posix()}\n"
            "exporters:\n- type: alpaca\n"
            "output_split: {train: 0.75, test: 0.25}\n"
            "output_split_group: group\n",
            encoding="utf-8",
        )
        out = tmp_path / "out"
        result = CliRunner().invoke(app, ["run", str(tmp_path / "p.yaml"), "-o", str(out)])
        assert result.exit_code == 0, result.output
        seen = _groups_per_split(out, ["train", "test"])
        assert len(seen) == 40
        assert all(len(names) == 1 for names in seen.values())

    def test_duplicates_across_groups_warn(self, tmp_path):
        seed = tmp_path / "seed.jsonl"
        with open(seed, "w", encoding="utf-8") as f:
            for g in range(40):
                f.write(
                    json.dumps(
                        {
                            "instruction": "The very same prompt repeated under a different group id each time.",
                            "output": f"Answer number {g} is written out here in several plain words for you.",
                            "group": f"g{g}",
                        }
                    )
                    + "\n"
                )
        with pytest.warns(UserWarning, match="output_split leakage"):
            Curator(
                CuratorConfig(
                    dataset=str(seed),
                    output_dir=str(tmp_path / "out"),
                    export_formats=["alpaca"],
                    dedup=False,
                    output_split={"train": 0.5, "test": 0.5},
                    output_split_group="group",
                )
            ).run()

    @staticmethod
    def _near_duplicate_seed(path):
        """Every group's prompt differs from the others in one word: near, not exact, duplicates."""
        base = _words("shared").split()
        with open(path, "w", encoding="utf-8") as f:
            for g in range(40):
                words = list(base)
                words[7] = _words(("g", g), 1)
                f.write(
                    json.dumps(
                        {"instruction": " ".join(words), "output": _words(g), "group": f"g{g}"}
                    )
                    + "\n"
                )

    def test_the_near_duplicate_check_can_be_skipped_from_config(self, tmp_path):
        seed = tmp_path / "seed.jsonl"
        self._near_duplicate_seed(seed)

        def run(out, **kwargs):
            Curator(
                CuratorConfig(
                    dataset=str(seed),
                    output_dir=str(tmp_path / out),
                    export_formats=["alpaca"],
                    dedup=False,
                    output_split={"train": 0.5, "test": 0.5},
                    output_split_group="group",
                    **kwargs,
                )
            ).run()

        with pytest.warns(UserWarning, match=r"0 exact and [1-9]\d* near-duplicate"):
            run("default")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            run("skipped", output_split_near_dup_threshold=None)

    def test_the_yaml_key_reaches_the_check(self, tmp_path):
        from curatorkit.cli import app

        seed = tmp_path / "seed.jsonl"
        self._near_duplicate_seed(seed)
        (tmp_path / "p.yaml").write_text(
            f"readers:\n- type: jsonl\n  path: {seed.as_posix()}\n"
            "exporters:\n- type: alpaca\n"
            "output_split: {train: 0.5, test: 0.5}\n"
            "output_split_group: group\n"
            "output_split_near_dup_threshold: null\n",
            encoding="utf-8",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = CliRunner().invoke(
                app, ["run", str(tmp_path / "p.yaml"), "-o", str(tmp_path / "out")]
            )
        assert result.exit_code == 0, result.output
