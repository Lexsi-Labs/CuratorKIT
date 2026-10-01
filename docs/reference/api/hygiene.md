# Hygiene

Secrets detection, PII pseudonymisation and gating, toxicity, jailbreak and output-safety
filtering. Presidio, Detoxify and detect-secrets come from the
`hygiene` extra; PromptGuard (JailbreakGate) from the `safety` extra. PIIGate, JailbreakGate
and OutputSafetyGate fall back to regex/keyword heuristics when their model is not available.

::: curatorkit.hygiene
      options:
        show_source: false
        show_root_heading: true
        members_order: source
        separate_signature: true
        show_signature_annotations: true
        merge_init_into_class: true
