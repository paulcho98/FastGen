# Batch Inference + LatentSync Compositing — Design Spec

**Date:** 2026-03-30
**Status:** Draft
**Extends:** `inference_causal.py` (causal OmniAvatar inference script)

## Goal

Extend `inference_causal.py` with:
1. Batch inference over a directory of samples (model loaded once)
2. LatentSync compositing (face detection → align → generate → paste back)
3. Face detection caching (one-time per video, stored to disk)

## New CLI Arguments

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--input_dir` | str | None | Directory of sample subdirs. Mutually exclusive with `--video_path`. Each subdir has `sub_clip.mp4`, `audio.wav`, optionally `prompt.txt` and precomputed `.pt` files. |
| `--output_dir` | str | None | Output directory for batch mode. One video per sample. |
| `--latentsync` | flag | False | Enable face detection + 512x512 alignment + compositing pipeline. |
| `--face_cache_dir` | str | None | Directory for `{stem}_face_cache.pt` files. Required when `--latentsync`. |
| `--use_mouth_only` | flag | True | When compositing, blend only mouth region (keep original upper face). |
| `--skip_existing` | flag | False | Skip samples whose output already exists (for resume). |

## Argument Validation

- `--input_dir` and `--video_path` are mutually exclusive. One must be provided.
- `--input_dir` requires `--output_dir`.
- `--latentsync` requires `--face_cache_dir`.
- When `--latentsync` is set, input video does NOT need to be 512x512 (arbitrary resolution accepted).
- When `--latentsync` is NOT set, input must be 512x512 (current behavior, error otherwise).

## Batch Flow

```
1. Parse args, validate
2. Load models once:
   - CausalOmniAvatarWan (1.3B student)
   - VAE (for encode + decode)
   - Wav2Vec + T5 (unless all samples have precomputed tensors)
   - ImageProcessor (only when --latentsync)

3. Enumerate samples:
   - If --input_dir: list subdirs, each is a sample
   - If --video_path: single sample (current behavior)

4. For each sample:
   a. Check skip_existing
   b. Resolve audio (from subdir audio.wav or --audio_path)
   c. Compute generation length from audio
   d. If --latentsync:
      - preprocess_with_latentsync() → face_cache + aligned 512x512 frames
      - Use aligned frames as reference video input
   e. Build conditioning (from raw encoding or precomputed .pt files)
   f. run_inference() — AR loop (UNCHANGED)
   g. If --latentsync:
      - VAE decode → float tensor [T, C, H, W] in [0, 1]
      - composite_with_latentsync_float() → original-res composited numpy
      - Save composited video + audio
      - Save aligned 512x512 video + audio (for metrics)
   h. Else:
      - decode_and_save() (current behavior)
   i. Clear model caches, log progress

5. Print summary (total, skipped, failed, succeeded)
```

## LatentSync Pipeline

### Preprocessing (before generation)

Reuse `preprocess_with_latentsync()` from `inference_v2v.py` (lines 92-191):
1. Load video frames at original resolution
2. For each frame: `ImageProcessor.affine_transform(frame)` → aligned face [C, 512, 512], box, affine_matrix
3. Cache to `{face_cache_dir}/{stem}_face_cache.pt`
4. On subsequent runs: load from cache if resolution matches, skip detection

Face cache format (compatible with existing caches from other scripts):
```python
{
    "aligned_faces": list of [C, H, W] uint8 tensors,
    "boxes": list of (x1, y1, x2, y2) or None,
    "affine_matrices": list of [1, 2, 3] affine tensors,
    "resolution": int (512),
    "num_frames": int,
}
```

### Generation Input

When `--latentsync` is enabled:
- The aligned 512x512 face crops become the reference video
- These are converted to tensor, masked, VAE-encoded as usual
- Audio encoding unchanged
- Conditioning dict unchanged — generation pipeline sees 512x512 input

### Post-processing (after generation)

Reuse `composite_with_latentsync_float()` from `inference_v2v.py` (lines 264-335):
1. VAE decode output latents → float tensor `[T, C, H, W]` in `[0, 1]`
2. Per frame:
   - Optional mouth-only blend: `generated * (1-mask) + original_aligned * mask`
   - Resize to bounding box dimensions
   - Convert `[0,1]` → `[-1,1]`
   - `restore_img()` — inverse affine + soft-blend paste onto original frame
3. Save composited frames as video + mux audio

### Outputs (per sample, when --latentsync)

| File | Description |
|------|-------------|
| `{stem}.mp4` | Composited original-resolution video with audio |
| `{stem}_aligned.mp4` | Aligned 512x512 video with audio (for metrics) |

### Outputs (per sample, without --latentsync)

| File | Description |
|------|-------------|
| `{stem}.mp4` | Generated 512x512 video with audio |

## Dependencies

LatentSync compositing requires:
- `insightface` + `onnxruntime` (face detection)
- `kornia` (affine transforms)
- These are imported only when `--latentsync` is set (lazy import)

The LatentSync utility module at `OmniAvatar-Train/OmniAvatar/utils/latentsync/` is imported
via the existing `OMNIAVATAR_ROOT` sys.path entry. No new module creation needed.

## Error Handling (Batch)

- Face detection failure on any frame → skip that sample, log warning, continue
- VAE/model error on a sample → skip, log error with traceback, continue
- Each sample wrapped in try/except for isolation
- Summary at end: N total, N succeeded, N skipped (existing), N failed

## What Does NOT Change

- `run_inference()` — the AR loop is completely unchanged
- `build_condition()` / `build_condition_from_precomputed()` — unchanged
- Model loading functions — unchanged
- Single-video mode (`--video_path`) — unchanged, fully backward compatible
