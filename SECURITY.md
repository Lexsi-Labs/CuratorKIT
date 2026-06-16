# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ |
| < 1.0   | ❌ |

## Reporting a vulnerability

Please **do not open a public issue** for security vulnerabilities. Report them
privately to **hello@lexsi.ai** (or via GitHub's private vulnerability reporting on
this repository). Include a description, reproduction steps, and the affected
version. You can expect an acknowledgement within **5 business days**.

## Scope notes for the hygiene gates

CuratorKIT's hygiene module (secrets detection, PII pseudonymization, toxicity
filtering) is a best-effort filter, not a guarantee. Detection gaps, meaning patterns
the gates miss, are quality issues, and we welcome them as regular GitHub issues,
**unless** your report would expose real credentials or personal data, in which case
report privately as above. Never paste real secrets or PII into a public issue,
even as a reproduction case.

API keys for LLM backends are read from environment variables and are never written
to manifests, dataset cards, or logs by design. If you find a code path that leaks
one, that is a security vulnerability. Report it privately.
