# LLM backends

Backends used by generation tasks and LLM-judge gates. The LiteLLM backend reaches
any provider LiteLLM supports and requires the `generation` extra. The Ollama backend
talks to a local server using only the standard library, so it works on a core
install.

::: curatorkit.llm
      options:
        show_source: false
        show_root_heading: true
        members_order: source
        separate_signature: true
        show_signature_annotations: true
        merge_init_into_class: true
