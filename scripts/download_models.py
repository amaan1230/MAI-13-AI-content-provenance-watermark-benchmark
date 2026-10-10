"""Pre-download every model used by the shipped experiment configs, so runs can use HF_OFFLINE=true.

python scripts/download_models.py            # all models
python scripts/download_models.py --minimal  # generator + scoring + embedding only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = [
    # (repo id, kind, purpose)
    ("Qwen/Qwen2.5-0.5B", "causal", "generator (watermarked + unwatermarked text)"),
    ("gpt2", "causal", "scoring / observer LM for AI-text detectors"),
    ("distilgpt2", "causal", "performer LM for Binoculars"),
    ("gpt2-medium", "causal", "oracle LM for perplexity"),
    ("sentence-transformers/all-MiniLM-L6-v2", "sbert", "semantic similarity"),
    ("distilroberta-base", "mlm", "lexical substitution attack"),
    ("Helsinki-NLP/opus-mt-en-de", "seq2seq", "back-translation attack (en->de)"),
    ("Helsinki-NLP/opus-mt-de-en", "seq2seq", "back-translation attack (de->en)"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "causal", "local paraphrase attack (hf_local provider)"),
]
MINIMAL = {"Qwen/Qwen2.5-0.5B", "gpt2", "distilgpt2", "sentence-transformers/all-MiniLM-L6-v2"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimal", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import list_repo_files, snapshot_download

    failures = 0
    for repo, _kind, purpose in MODELS:
        if args.minimal and repo not in MINIMAL:
            continue
        t0 = time.time()
        try:
            patterns = ["*.json", "*.txt", "*.model", "*.spm", "*.tiktoken", "merges.txt", "vocab.*", "*.py"]
            files = list_repo_files(repo)
            # Prefer safetensors; some repos (e.g. Helsinki-NLP/opus-mt-*) only publish .bin weights.
            if any(f.endswith(".safetensors") for f in files):
                patterns.append("*.safetensors")
            else:
                patterns.append("pytorch_model*.bin")
            snapshot_download(repo, allow_patterns=patterns)
            print(f"OK   {repo:45s} {time.time() - t0:6.1f}s  ({purpose})", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {repo:45s} {type(exc).__name__}: {exc}", flush=True)
    if failures:
        print(f"\n{failures} model(s) failed. On Windows, error 1314 means: enable Developer Mode "
              "(Settings > System > For developers) or run PowerShell as Administrator, then re-run.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
