# Contributing

The canonical contribution guide lives in the repository:
[CONTRIBUTING.md](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/CONTRIBUTING.md).
This page summarises it.

## Development setup

```bash
git clone https://github.com/Lexsi-Labs/CuratorKIT.git
cd CuratorKIT
pip install -e ".[dev]"
pre-commit install
```

Both checks must pass before a PR can merge:

```bash
ruff check .            # lint
pytest tests/unit -q    # unit tests; run without heavy extras, external deps are mocked
```

Strict mypy is configured and adopted incrementally; it is encouraged, not gating.

## Adding a component

Every pipeline stage implements one of four small ABCs in `curatorkit/interfaces.py`:

| ABC | Contract | Implementations |
|-----|----------|-----------------|
| `BaseReader` | `read() -> (samples, rejected)` | `curatorkit/connectors/` |
| `BaseGate` | `run(samples) -> (passed, rejected)` | `curatorkit/gates/`, `curatorkit/hygiene/` |
| `BaseNormalizer` | `run(samples) -> samples` | `curatorkit/normalizers/` |
| `BaseExporter` | `export(samples, output_dir)` | `curatorkit/exporters/` |

Read the [architecture reference](../reference/architecture.md) before starting. A
complete change includes the implementation, wiring through `CuratorConfig` or the
YAML config, unit tests, and an update to the matching guide, in particular the
[configuration reference](../reference/configuration.md) for any `CuratorConfig`
change.

## Conventions

- PR titles follow Conventional Commits (`feat:`, `fix:`, `docs:`, …).
- Behaviour changes need tests.
- Add a changelog entry under the *Unreleased* heading.
- Docs pages use British spelling; keep each file internally consistent.

## Where to ask

Questions go to [GitHub Discussions](https://github.com/Lexsi-Labs/CuratorKIT/discussions);
bugs to [Issues](https://github.com/Lexsi-Labs/CuratorKIT/issues). Security issues are
never filed publicly; see
[SECURITY.md](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/SECURITY.md).
Participation is governed by the
[Code of Conduct](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/CODE_OF_CONDUCT.md).
