# Data models

Every sample flowing through a pipeline is a `DataSample`; rejections are wrapped in
`RejectedSample`; each step appends an immutable `ProvenanceRecord`.

::: curatorkit.schema.DataSample
      options:
        show_source: false
        show_root_heading: true
        members_order: source
        separate_signature: true
        show_signature_annotations: true
        merge_init_into_class: true

::: curatorkit.schema.RejectedSample
      options:
        show_source: false
        show_root_heading: true
        members_order: source
        separate_signature: true
        show_signature_annotations: true
        merge_init_into_class: true

::: curatorkit.schema.ProvenanceRecord
      options:
        show_source: false
        show_root_heading: true
        members_order: source
        separate_signature: true
        show_signature_annotations: true
        merge_init_into_class: true
