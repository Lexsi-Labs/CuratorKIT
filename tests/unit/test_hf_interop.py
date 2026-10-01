"""Export folder = Hugging Face dataset folder + lexsi provenance + TRL consumer smoke.

Every export config must load with `datasets.load_dataset(output_dir, config)`,
manifest/rejected/provenance files must never become splits, and the chat and
preference configs must train as-is in TRL (tiny random Cohere models, CPU,
no downloads).
"""

import importlib.metadata
import json
import re
import warnings

import pytest

datasets = pytest.importorskip("datasets")

import curatorkit  # noqa: E402
from curatorkit.curator import Curator, CuratorConfig  # noqa: E402
from curatorkit.exporters import EXPORTERS  # noqa: E402
from curatorkit.manifest import DatasetCardGenerator, ProvenanceManifest  # noqa: E402
from curatorkit.pipeline import PipelineResult  # noqa: E402
from curatorkit.schema import DataSample  # noqa: E402

ALL_CONFIGS = {"sft_alpaca", "sft_sharegpt", "sft_messages", "dpo", "grpo", "ppo", "corpus"}


def _samples(n=4):
    out = []
    for i in range(n):
        out += [
            DataSample(source_uri="t", instruction=f"Question {i} about rivers?", input=f"context {i}",
                       output=f"Rivers carry water {i}."),
            DataSample(source_uri="t", task_type="conversational", instruction=f"Hi {i}",
                       output=f"Hello {i}", metadata={
                           "system_prompt": "Be brief.",
                           "turns": [{"role": "user", "content": "More?"},
                                     {"role": "assistant", "content": f"Sure {i}."}]}),
            DataSample(source_uri="t", task_type="preference", instruction=f"Name a colour {i}.",
                       chosen=f"Blue {i}.", rejected="No."),
            DataSample(source_uri="t", task_type="grpo", instruction=f"Add {i} and 1.",
                       responses=[str(i + 1), "0"], reward_scores=[1.0, 0.0]),
            DataSample(source_uri="t", task_type="prompt_only", instruction=f"Write about {i}."),
            DataSample(source_uri="t", task_type="language_modeling", output=f"Chunk {i} text."),
        ]
    return out


def _export(samples, out, splits):
    for split in splits or [None]:
        d = out / split if split else out
        d.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():  # each exporter warns about the task types it skips
            warnings.simplefilter("ignore")
            for cls in EXPORTERS.values():
                cls().export(samples, d)
    m = ProvenanceManifest(PipelineResult(passed=samples, rejected=[]), "h", out)
    m.write()
    m.write_rejected_sidecar()
    DatasetCardGenerator().generate(m.build(), out)
    return m


@pytest.mark.parametrize("splits", [None, ["train", "validation"]])
def test_every_config_round_trips(tmp_path, splits):
    samples = _samples()
    _export(samples, tmp_path, splits)
    readme = (tmp_path / "README.md").read_text()
    assert readme.startswith("---\n") and "configs:" in readme

    expected_splits = set(splits or ["train"])
    for cfg in ALL_CONFIGS:
        ds = datasets.load_dataset(str(tmp_path), cfg)
        assert set(ds) == expected_splits, cfg
        rows = (tmp_path / ("train" if splits else "") / f"{cfg}.jsonl").read_text().count("\n")
        assert len(ds["train"]) == rows > 0, cfg
        assert f"`{cfg}`" in readme  # column schema documented per config

    messages = datasets.load_dataset(str(tmp_path), "sft_messages", split="train")
    assert messages.column_names == ["messages"]
    conv = messages[1]["messages"]
    assert [m["role"] for m in conv] == ["system", "user", "assistant", "user", "assistant"]
    assert set(conv[0]) == {"role", "content"}
    assert messages[0]["messages"][0] == {"role": "user", "content": "Question 0 about rivers?\n\ncontext 0"}

    # Same turns in real ShareGPT layout.
    sharegpt = datasets.load_dataset(str(tmp_path), "sft_sharegpt", split="train")
    assert sharegpt.column_names == ["conversations"]
    conv = sharegpt[1]["conversations"]
    assert [t["from"] for t in conv] == ["system", "human", "gpt", "human", "gpt"]
    assert set(conv[0]) == {"from", "value"}
    assert sharegpt[0]["conversations"][0] == {"from": "human", "value": "Question 0 about rivers?\n\ncontext 0"}

    alpaca = datasets.load_dataset(str(tmp_path), "sft_alpaca", split="train")
    assert alpaca.column_names == ["instruction", "input", "output"]
    dpo = datasets.load_dataset(str(tmp_path), "dpo", split="train")
    assert dpo.column_names == ["prompt", "chosen", "rejected"] and len(dpo) == 4

    # No config name -> the default config; non-data files never become data.
    default = datasets.load_dataset(str(tmp_path))
    assert set(default) == expected_splits
    assert default["train"].column_names == ["instruction", "input", "output"]


def test_empty_export_is_not_a_config(tmp_path):
    _export(_samples()[:1], tmp_path, None)  # one alpaca-type sample: dpo.jsonl is empty
    assert (tmp_path / "dpo.jsonl").stat().st_size == 0
    assert "config_name: dpo" not in (tmp_path / "README.md").read_text()
    assert set(datasets.load_dataset(str(tmp_path), "sft_alpaca")) == {"train"}


def test_rerun_into_same_folder_is_not_served_from_cache(tmp_path):
    # datasets caches a local folder by its README YAML; the per-config sha256 must bust it.
    for n in (2, 3):
        _export(_samples(n), tmp_path / "out", None)
        assert len(datasets.load_dataset(str(tmp_path / "out"), "dpo", split="train")) == n


def _seed(path, n=20):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "instruction": f"Explain topic number {i} in a few clear and simple sentences please.",
                "output": f"Topic {i} is explained here in several plain words for the reader to follow.",
            }) + "\n")


def test_curator_run_writes_loadable_folder_and_provenance(tmp_path):
    parent = {"schema": "lexsi.provenance/1", "library": "upstream", "version": "9"}
    (tmp_path / "src").mkdir()
    seed = tmp_path / "src" / "seed.jsonl"
    _seed(seed)
    (tmp_path / "src" / "lexsi_provenance.json").write_text(json.dumps(parent))
    out = tmp_path / "out"
    result = Curator(CuratorConfig(
        dataset=str(seed), output_dir=str(out),
        export_formats=["alpaca", "sharegpt", "dpo"],
        output_split={"train": 0.8, "validation": 0.2},
    )).run()
    assert len(result.passed) == 20

    prov = json.loads((out / "lexsi_provenance.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["provenance"] == prov
    assert prov["schema"] == "lexsi.provenance/1" and prov["library"] == "curatorkit"
    assert prov["version"] == curatorkit.__version__ == importlib.metadata.version("curatorkit")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", prov["created_at"])
    assert prov["base_model"] is None and prov["method"] == "curate"
    assert prov["inputs"] == [{"kind": "dataset", "ref": str(seed), "config": None, "provenance": parent}]
    # No generation task and no judge gate ran, so no model is claimed.
    assert prov["params"]["generator_model"] is None and prov["params"]["judge_model"] is None
    assert "lexsi_provenance.json" in (out / "checksums.txt").read_text()

    for cfg in ("sft_alpaca", "sft_sharegpt"):
        ds = datasets.load_dataset(str(out), cfg)
        assert set(ds) == {"train", "validation"}
        assert len(ds["train"]) == 16 and len(ds["validation"]) == 4
    with pytest.raises(ValueError):
        datasets.load_dataset(str(out), "dpo")  # nothing to export -> not a config


def test_provenance_records_generator_and_judge(tmp_path):
    pytest.importorskip("litellm")
    from curatorkit.manifest import PROVENANCE_FILE
    cfg = CuratorConfig(dataset={"name": "org/data", "subset": "en"}, output_dir=str(tmp_path),
                        llm_model="openai/gen", judge_llm_model="openai/judge", generation_task="qa",
                        reward_threshold=0.5)
    curator = Curator(cfg)
    steps = curator._build_steps(include_exporters=False)  # the LLM backends that would run
    curator._write_provenance(PipelineResult(passed=[], rejected=[]), tmp_path, steps)
    prov = json.loads((tmp_path / PROVENANCE_FILE).read_text())
    assert prov["method"] == "qa"
    assert prov["params"]["generator_model"] == "openai/gen"
    assert prov["params"]["judge_model"] == "openai/judge"
    assert prov["params"]["export_formats"] == ["alpaca", "sharegpt", "messages"]  # auto for qa
    assert prov["inputs"] == [{"kind": "dataset", "ref": "org/data", "config": "en", "provenance": None}]


@pytest.mark.parametrize("filename", ["sft_messages.jsonl", "sft_sharegpt.jsonl"])
def test_chat_export_reingests(tmp_path, filename):
    from curatorkit.connectors.jsonl import JSONLReader
    _export(_samples(), tmp_path, None)
    samples, _ = JSONLReader(tmp_path / filename).read()
    assert samples and all(s.task_type == "conversational" for s in samples)


# ── TRL consumer smoke: the folder trains as-is ──────────────────────────────

def _tiny(arch, texts):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]", t)})
    vocab = {w: i for i, w in enumerate(["<pad>", "<unk>", "<bos>", "<eos>", *words])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="<pad>", unk_token="<unk>", bos_token="<bos>", eos_token="<eos>")
    tok.chat_template = (
        "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} <eos> {% endfor %}"
        "{% if add_generation_prompt %}assistant : {% endif %}")
    small = dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                 num_attention_heads=4, num_key_value_heads=2, pad_token_id=0, bos_token_id=2,
                 eos_token_id=3)
    torch.manual_seed(0)
    if arch == "cohere":
        model = transformers.CohereForCausalLM(transformers.CohereConfig(**small))
    else:
        model = transformers.Cohere2ForCausalLM(transformers.Cohere2Config(
            **small, sliding_window=16, layer_types=["sliding_attention", "full_attention"]))
    return model, tok


def _all_text(ds):
    return [json.dumps(r) for r in ds]


@pytest.mark.parametrize("arch", ["cohere", "cohere2"])
def test_trl_sft_trains_on_sft_messages(tmp_path, arch):
    trl = pytest.importorskip("trl")
    _export(_samples(), tmp_path / "data", None)
    ds = datasets.load_dataset(str(tmp_path / "data"), "sft_messages", split="train")
    model, tok = _tiny(arch, _all_text(ds))
    args = trl.SFTConfig(output_dir=str(tmp_path / "sft"), max_steps=1, per_device_train_batch_size=2,
                         max_length=64, report_to="none", save_strategy="no", bf16=False, use_cpu=True)
    trainer = trl.SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    loss = trainer.train().training_loss
    assert loss == loss and loss > 0  # finite, not NaN


@pytest.mark.parametrize("arch", ["cohere", "cohere2"])
def test_trl_dpo_trains_on_dpo(tmp_path, arch):
    trl = pytest.importorskip("trl")
    _export(_samples(), tmp_path / "data", None)
    ds = datasets.load_dataset(str(tmp_path / "data"), "dpo", split="train")
    model, tok = _tiny(arch, _all_text(ds))
    model.save_pretrained(tmp_path / "tiny")  # DPOTrainer loads its reference model by path
    model = str(tmp_path / "tiny")
    args = trl.DPOConfig(output_dir=str(tmp_path / "dpo"), max_steps=1, per_device_train_batch_size=2,
                         max_length=64, report_to="none", save_strategy="no", bf16=False, use_cpu=True)
    trainer = trl.DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    loss = trainer.train().training_loss
    assert loss == loss and loss > 0


def test_cli_run_writes_same_folder(tmp_path):
    from typer.testing import CliRunner

    from curatorkit.cli import app
    seed = tmp_path / "seed.jsonl"
    _seed(seed)
    (tmp_path / "p.yaml").write_text(
        f"readers:\n- type: jsonl\n  path: {seed}\n"
        "exporters:\n- type: alpaca\n- type: sharegpt\n"
        "output_split: {train: 0.75, validation: 0.25}\n")
    out = tmp_path / "out"
    res = CliRunner().invoke(app, ["run", str(tmp_path / "p.yaml"), "-o", str(out)])
    assert res.exit_code == 0, res.output
    prov = json.loads((out / "lexsi_provenance.json").read_text())
    assert json.loads((out / "manifest.json").read_text())["provenance"] == prov
    assert prov["inputs"][0]["ref"] == str(seed) and prov["params"]["export_formats"] == ["alpaca", "sharegpt"]
    ds = datasets.load_dataset(str(out), "sft_sharegpt")
    assert (len(ds["train"]), len(ds["validation"])) == (15, 5)


# ── README.md ownership: a user's own README is never overwritten ────────────

def test_user_readme_is_left_untouched(tmp_path):
    (tmp_path / "README.md").write_text("# My project\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="not written by curatorkit"):
        _export(_samples(), tmp_path, None)
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# My project\n"
    assert (tmp_path / "dataset_card.md").exists()


def test_curatorkit_readme_is_rewritten_on_rerun(tmp_path):
    _export(_samples(2), tmp_path, None)
    readme = tmp_path / "README.md"
    lf = readme.read_bytes().replace(b"\r\n", b"\n")
    for data in (lf, lf.replace(b"\n", b"\r\n")):  # LF and CRLF (e.g. a Windows checkout)
        readme.write_bytes(data)
        assert DatasetCardGenerator._readme_is_ours(readme)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        m = ProvenanceManifest(PipelineResult(passed=_samples(3), rejected=[]), "h", tmp_path)
        DatasetCardGenerator().generate(m.build(), tmp_path)
    assert "configs:" in readme.read_text(encoding="utf-8")


def test_write_readme_false_skips_readme(tmp_path):
    m = _export(_samples(), tmp_path / "a", None)
    (tmp_path / "b").mkdir()
    DatasetCardGenerator().generate(m.build(), tmp_path / "b", write_readme=False)
    assert (tmp_path / "b" / "dataset_card.md").exists()
    assert not (tmp_path / "b" / "README.md").exists()


def test_curator_write_hf_readme_false(tmp_path):
    seed = tmp_path / "seed.jsonl"
    _seed(seed)
    out = tmp_path / "out"
    Curator(CuratorConfig(dataset=str(seed), output_dir=str(out), export_formats=["messages"],
                          write_hf_readme=False)).run()
    assert (out / "sft_messages.jsonl").exists() and (out / "dataset_card.md").exists()
    assert not (out / "README.md").exists()


def test_cli_write_hf_readme_false(tmp_path):
    from typer.testing import CliRunner

    from curatorkit.cli import app
    seed = tmp_path / "seed.jsonl"
    _seed(seed)
    (tmp_path / "p.yaml").write_text(
        f"readers:\n- type: jsonl\n  path: {seed}\n"
        "exporters:\n- type: messages\n"
        "write_hf_readme: false\n")
    out = tmp_path / "out"
    res = CliRunner().invoke(app, ["run", str(tmp_path / "p.yaml"), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "sft_messages.jsonl").exists()
    assert not (out / "README.md").exists()
