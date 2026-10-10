# datasets/

Source records are **materialised on demand** by experiments (C4 is streamed from the Hugging Face Hub)
and cached here as `<name>/records-<hash>/records.jsonl` with a `records.meta.json` (SHA-256) and an
immutable `manifest-<version>.json`. These generated files are git-ignored; they are reproducible from
the experiment config.

`manifests/` holds copies of the manifests for the published experiments, so the exact dataset
versions (records hash, split seed, split counts, leakage audit) are documented in the repository.
See `docs/DATASET_CARD.md`.
