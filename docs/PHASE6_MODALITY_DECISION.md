# Phase 6 decision: image and audio modalities

**Status:** Not implemented. **Decision:** defer; the approved deliverable is the text pipeline.

## Assessment

The project brief asks to "start with a fully implemented text-modality pipeline", use "a modular
architecture that supports image and audio watermarking extensions", and **not claim multimodal
functionality until implementations and experiments exist**. Phase 6 asks whether the approved scope
*requires* image/audio.

| Criterion | Text (done) | Image / audio (if added) |
|---|---|---|
| Watermark methods | KGW, Unigram, zero-width | Would need e.g. Tree-Ring / StegaStamp (image), AudioSeal (audio) — large models, mostly GPU-oriented |
| Attacks | 9 implemented | JPEG, crop, resize, blur, noise, regeneration; MP3/Opus, resampling, time-stretch, noise |
| Quality metrics | semantic similarity, perplexity | PSNR, SSIM, LPIPS / PESQ, STOI, ViSQOL |
| Data | C4 (ODC-BY) | e.g. MS-COCO (CC-BY 4.0 annotations; image licences vary), LibriSpeech (CC-BY 4.0) |
| Compute on target hardware (CPU laptop) | ~2–3 h for the main benchmark | Diffusion-based image watermarking is impractical on CPU |

Adding a modality properly means a new watermark family, attack family, quality metrics, dataset card
and an executed experiment — roughly the size of the text work again. A stub would violate the brief's
"no unverified claims" rule, so none was added.

## What is already in place for an extension

- `content_items.modality` column (default `text`) and an artifact store for binary payloads.
- Modality-agnostic interfaces: `WatermarkMethod`, `Attack`, `AITextDetector` (rename to `Detector`),
  `ArtifactStore`; the evaluation metrics module is score-based and modality-independent.
- The runner's stages are generic except for generation (text LM) and quality (text metrics).

## Recommended path if scope is expanded

1. Image first: `backend/watermark/image/` with a post-hoc DWT-DCT baseline (CPU-feasible, e.g. the
   `invisible-watermark` package) and a learned method if a GPU is available; attacks via Pillow.
2. Generalise `Sample.text` to a `payload_ref` and add per-modality quality functions.
3. Run a robustness experiment with the same statistical protocol (validation thresholds, held-out test,
   CIs) before claiming support.
