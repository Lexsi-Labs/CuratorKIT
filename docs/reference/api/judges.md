# Judges

Rubric LLM-as-judge scorers (Prometheus 2, G-Eval / DeepEval, FActScore) and the gates built on them. They call any `BaseLLM`, so they need the `generation` extra but no judge-specific package.

::: curatorkit.judges
      options:
        show_source: false
        show_root_heading: true
        members_order: source
        separate_signature: true
        show_signature_annotations: true
        merge_init_into_class: true
