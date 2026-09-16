"""
Shared prompt-template placeholder validation.

Used by generation tasks, quality gates, and the reward-refiner/diagnostic-
probe recovery modules whenever a caller supplies a custom prompt template.

str.format(**kwargs) silently ignores a kwarg with no matching placeholder in
the template — so a template missing a required variable doesn't crash, it
just drops that data from the actual prompt sent to the LLM. Every caller of
validate_prompt_template() does so at construction time, so a broken custom
template fails fast with a clear error before any LLM calls, instead of
silently generating the same prompt for every sample in the run.
"""

from __future__ import annotations


def validate_prompt_template(
    template: str, required_vars: list[str], name: str = "prompt_template"
) -> None:
    """Raise ValueError if any required {var} placeholder is absent from template."""
    missing = [v for v in required_vars if "{" + v + "}" not in template]
    if missing:
        raise ValueError(
            f"Custom {name} is missing required placeholder(s): "
            + ", ".join(f"{{{v}}}" for v in missing)
            + f". Available: {', '.join('{' + v + '}' for v in required_vars)}"
        )
