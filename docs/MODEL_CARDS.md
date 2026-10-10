# Model and method cards

All models are open checkpoints from the Hugging Face Hub, chosen so the full benchmark runs on a
CPU-only laptop. Exact resolved library versions are in `requirements.lock`; model revisions are the
Hub `main` revisions at download time (set `revision` in configs to pin a commit).

## Pretrained models used

| Role | Model | Licence | Notes |
|---|---|---|---|
| Generator (watermarked/unwatermarked) | `Qwen/Qwen2.5-0.5B` (base) | Apache-2.0 | Continuation of C4 prompts; float32 on CPU. |
| Local paraphraser (attack) | `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 | Weak paraphraser; a lower bound on paraphrase strength. |
| Observer / scoring LM | `gpt2` | MIT | Detector features; different family from the generator (no white-box access). |
| Performer LM (Binoculars) | `distilgpt2` | Apache-2.0 | Shares GPT-2 tokenizer with observer. |
| Oracle LM (perplexity) | `gpt2-medium` | MIT | Quality metric only; differs from generator. |
| Sentence embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | Semantic similarity of attacked vs. original text. |
| Masked LM (attack) | `distilroberta-base` | Apache-2.0 | Lexical substitution. |
| Translation (attack) | `Helsinki-NLP/opus-mt-en-de`, `opus-mt-de-en` | CC-BY-4.0 | Greedy back-translation via German. |
| Optional remote | Gemini (`GEMINI_MODEL`) | Google terms | Paraphrase/translation/generation when `GEMINI_API_KEY` is set. |

## Watermark methods

### KGW (`kgw`, v1.0.0) — Kirchenbauer et al., ICML 2023
- **Embedding:** at each step, a green list of γ·|V| tokens is derived from the previous `h` tokens and
  the secret key; δ is added to green logits before temperature/top-p.
- **Detection:** re-tokenise, count green tokens among T scored tokens (unique (context, token) pairs),
  z = (g − γT)/√(Tγ(1−γ)); positive if z ≥ threshold (default 4.0, one-sided p ≈ 3·10⁻⁵).
- **Defaults:** γ=0.25, δ=2.0, h=1, min_tokens=16.
- **Implementation note:** green lists use a keyed splitmix64 PRF instead of `torch.randperm`
  (torch-version-independent); exact γ-fraction holds in expectation, verified in tests.
- **Assumptions:** detector has the same tokenizer and key; text has enough entropy.
- **Limitations:** short or low-entropy text; paraphrase/translation re-sample tokens; key compromise allows spoofing/scrubbing.

### Unigram (`unigram`, v1.0.0) — Zhao et al., ICLR 2024
- KGW with h=0: one fixed green list. More robust to local edits (a substitution changes only the
  edited token's status), but the fixed list can be estimated from enough watermarked text, and repeated
  tokens are counted once, so fewer tokens are scored.

### Zero-width (`zero_width`, v1.0.0) — simulation baseline
- Post-hoc: a key-derived 16-bit payload encoded as U+200B/U+200C after every 8th word.
- **Purpose:** demonstrates format-level marking: survives word edits, destroyed by Unicode
  normalisation, trivially spoofable by copying the marks. Not a recommended provenance mechanism.

## AI-text detectors

### `tfidf_lr` (baseline)
Word 1–2-gram + char_wb 3–5-gram TF-IDF (50k features each), L2 logistic regression with balanced class
weights. Trained on the train split, human vs. *unwatermarked* AI text.

### `lm_features_lr` (advanced)
Seven features under the observer LM (mean/std token log-prob, mean log-rank, mean entropy, top-1 and
top-10 rank fractions, Binoculars ratio with the performer), standardised, logistic regression.
Coefficients are reported for interpretability.

### `binoculars` (zero-shot)
Score = −(log-PPL_observer / cross-PPL(observer, performer)) after Hans et al. (2024). No training;
threshold still chosen on validation data. With GPT-2-sized observer/performer this is a weak
instantiation; results are **not** comparable to the paper's Falcon-7B setting.

### Intended use and out-of-scope use
- Intended: benchmarking, robustness research, teaching.
- Out of scope: decisions about individuals (academic integrity cases, hiring, moderation) — error
  rates are measured on one domain and one generator and do not transfer.
