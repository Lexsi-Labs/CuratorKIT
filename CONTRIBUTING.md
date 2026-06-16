# Contributing to CuratorKIT

Thank you for your interest in improving CuratorKIT! Contributions of all kinds are
welcome: bug reports, documentation fixes, new connectors, generation tasks, quality
gates, and exporters.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development setup

CuratorKIT requires Python 3.11 or newer.

```bash
git clone https://github.com/Lexsi-Labs/CuratorKIT.git
cd CuratorKIT
pip install -e ".[dev]"
```

The `dev` extra installs pytest, ruff, mypy, and the packaging tools. If your change
touches an optional-dependency area, install the matching extra as well. For example, use
`pip install -e ".[dev,connectors]"` for Parquet/HuggingFace connector work, or
`pip install -e ".[dev,hygiene]"` for the hygiene gates (PII work also needs
`python -m spacy download en_core_web_sm`).

## Running the checks

Both must pass before a PR can merge:

```bash
ruff check .            # lint (rules configured in pyproject.toml)
pytest tests/unit -q    # unit tests (run without heavy extras; external deps are mocked)
```

A strict mypy config ships in `pyproject.toml` and is being adopted incrementally;
running `mypy curatorkit` on changed files is encouraged but not yet gating.

We also ship a [pre-commit](https://pre-commit.com) config that runs ruff, basic
hygiene checks, and secret scanning on every commit. Install the hooks once after
cloning (`pre-commit` comes with the `dev` extra):

```bash
pre-commit install
```

Notes on tests:

- `tests/unit/` is the default suite and runs with only the `dev` extra installed.
- `tests/integration/` (e.g. `test_trl.py`) needs the `trl` extra and a large
  torch/transformers stack. It is not expected for routine PRs.
- Optional-dependency tests are tagged with the markers declared in `pyproject.toml`
  (`integration`, `slow`, `hf`, `parquet`, `generation`, `embedding`); select or skip
  them with `pytest -m`.

## Architecture: how to add a component

CuratorKIT is plugin-style: every pipeline stage implements one of four small ABCs in
`curatorkit/interfaces.py`:

| ABC | Contract | Where implementations live |
|-----|----------|---------------------------|
| `BaseReader` | `read() -> (samples, rejected)` | `curatorkit/connectors/` |
| `BaseGate` | `run(samples) -> (passed, rejected)` | `curatorkit/gates/`, `curatorkit/hygiene/` |
| `BaseNormalizer` | `run(samples) -> samples` | `curatorkit/normalizers/` |
| `BaseExporter` | `export(samples, output_dir)` | `curatorkit/exporters/` |

New connectors should subclass `BaseConnector` in `curatorkit/connectors/base.py` and
implement only `_iter_rows()`; the base class handles preprocessing, field mapping,
format detection, and rejection tracking. New generation tasks subclass the generator
base in `curatorkit/generators/base.py`.

Read [docs/reference/architecture.md](docs/reference/architecture.md) before starting. It documents the
`DataSample`/`RejectedSample` contracts, the plugin interfaces, and the pipeline
orchestration in detail.

A typical "add a component" change includes:

1. The implementation, subclassing the relevant ABC.
2. Wiring: expose it through `CuratorConfig` (`curatorkit/curator.py` /
   `curatorkit/config.py`) and/or the YAML config if user-facing.
3. Unit tests in `tests/unit/`.
4. Documentation in the matching `docs/` guide.

## Documentation

The docs site is built with MkDocs Material:

```bash
pip install -e ".[docs]"
mkdocs serve      # live preview at http://127.0.0.1:8000
```

Docs pages use British spelling (customisation, normalisation); please keep each file
consistent. If your change adds or alters a `CuratorConfig` field, update
[docs/reference/configuration.md](docs/reference/configuration.md) in the same PR.

## Pull request conventions

- **Titles** follow Conventional Commits: `feat: ...`, `fix: ...`, `docs: ...`,
  `refactor: ...`, `test: ...`, `chore: ...`.
- **Behaviour changes need tests.** A PR that changes what the pipeline does without a
  test demonstrating the new behaviour will be sent back.
- **Keep docs in sync**, especially `docs/reference/configuration.md` for any
  `CuratorConfig` change, and `docs/reference/cli.md` for CLI/YAML changes.
- **Add a CHANGELOG entry** under the `Unreleased` heading in
  [CHANGELOG.md](CHANGELOG.md).
- Keep PRs focused: one logical change per PR is much faster to review.

## Reporting bugs and asking questions

See [SUPPORT.md](SUPPORT.md) for where to ask questions, report bugs, and reach the
maintainers. Security issues should never be filed publicly; see
[SECURITY.md](SECURITY.md).

---

CuratorKIT is maintained by [Lexsi Labs](https://lexsi.ai). We're glad you're here.
