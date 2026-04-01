# Batch Inference + LatentSync Compositing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `inference_causal.py` with batch inference over sample directories, LatentSync face detection/alignment/compositing, and face detection caching.

**Architecture:** Add new CLI args, LatentSync functions (copied verbatim from `inference_v2v.py` reference), and restructure `main()` to loop over samples. The AR inference loop (`run_inference`) is unchanged. LatentSync functions use the existing `OmniAvatar.utils.latentsync` module.

**Tech Stack:** Existing deps + `insightface`, `onnxruntime`, `kornia` (lazy-imported when `--latentsync`)

**Spec:** `docs/superpowers/specs/2026-03-30-batch-latentsync-design.md`

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `scripts/inference/inference_causal.py` | Modify | Add CLI args, LatentSync functions, batch loop in main() |

Single file modification — all new code added to the existing script.

---

### Task 1: Add New CLI Arguments

**Files:**
- Modify: `scripts/inference/inference_causal.py` (parse_args function)

- [ ] **Step 1: Add batch and LatentSync arguments to parse_args()**

In `parse_args()`, add these arguments after the existing `--precomputed_dir` argument:

```python
    # --- Batch inference ---
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of sample subdirs (each with sub_clip.mp4, audio.wav). "
                             "Mutually exclusive with --video_path.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for batch mode")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples whose output already exists (for resume)")

    # --- LatentSync compositing ---
    parser.add_argument("--latentsync", action="store_true",
                        help="Enable face detection + 512x512 alignment + compositing")
    parser.add_argument("--face_cache_dir", type=str, default=None,
                        help="Directory for face detection caches (required with --latentsync)")
    parser.add_argument("--use_mouth_only", action="store_true", default=True,
                        help="Blend only mouth region, keep original upper face (default: True)")
    parser.add_argument("--no_mouth_only", action="store_false", dest="use_mouth_only",
                        help="Composite entire generated face (disable mouth-only blending)")
```

- [ ] **Step 2: Add argument validation**

Add a `validate_args(args)` function after `parse_args()`:

```python
def validate_args(args):
    """Validate CLI argument combinations."""
    if args.input_dir is not None and args.video_path is not None:
        raise ValueError("--input_dir and --video_path are mutually exclusive")
    if args.input_dir is None and args.video_path is None:
        raise ValueError("Must provide either --input_dir or --video_path")
    if args.input_dir is not None and args.output_dir is None:
        raise ValueError("--input_dir requires --output_dir")
    if args.latentsync and args.face_cache_dir is None:
        raise ValueError("--latentsync requires --face_cache_dir")
    # In single-video mode, output_path is required
    if args.input_dir is None and args.output_path is None:
        raise ValueError("--video_path mode requires --output_path")
```

Also update `parse_args` to make `--video_path` and `--output_path` no longer required (they're optional when using `--input_dir`):

Change from:
```python
    parser.add_argument("--video_path", required=True, ...
    parser.add_argument("--output_path", required=True, ...
```
To:
```python
    parser.add_argument("--video_path", type=str, default=None, ...
    parser.add_argument("--output_path", type=str, default=None, ...
```

- [ ] **Step 3: Verify syntax and help**

Run:
```bash
python -c "import ast; ast.parse(open('scripts/inference/inference_causal.py').read())"
python scripts/inference/inference_causal.py --help
```

Expected: No errors, new arguments visible in help.

- [ ] **Step 4: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat: add batch + LatentSync CLI arguments"
```

---

### Task 2: Add LatentSync Functions

**Files:**
- Modify: `scripts/inference/inference_causal.py`

These functions are copied **verbatim** from the reference implementation at
`../../OmniAvatar-Train/scripts/inference_v2v.py` to ensure identical behavior,
face cache compatibility, and compositing quality.

- [ ] **Step 1: Add preprocess_with_latentsync function**

Add this function after the existing `build_condition_from_precomputed()` function
(before the `# Inference & post-processing` section). Copy verbatim from
`inference_v2v.py` lines 92-191:

```python
def preprocess_with_latentsync(video_path, image_processor, face_detection_cache_dir, num_frames=81):
    """Detect faces, align to 512x512 via affine transform, with caching.

    Copied verbatim from OmniAvatar-Train/scripts/inference_v2v.py to ensure
    identical face cache format and detection behavior.

    Args:
        video_path: path to input video (arbitrary resolution)
        image_processor: ImageProcessor instance (from OmniAvatar.utils.latentsync)
        face_detection_cache_dir: directory for {stem}_face_cache.pt files
        num_frames: number of frames to process

    Returns:
        dict with video_path, original_frames, num_frames, aligned_faces,
        boxes, affine_matrices, detection_failures. Or None on failure.
    """
    if not os.path.exists(video_path):
        print(f"[LatentSync] Video not found: {video_path}")
        return None

    try:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        if video_basename in ("sub_clip", "video"):
            video_stem = os.path.basename(os.path.dirname(video_path))
        else:
            video_stem = video_basename
        face_cache_path = os.path.join(face_detection_cache_dir, f"{video_stem}_face_cache.pt")

        face_cache_loaded = False
        original_frames = None
        if os.path.isfile(face_cache_path):
            try:
                face_cache = torch.load(face_cache_path, weights_only=False)
                if face_cache.get("resolution") == image_processor.resolution:
                    boxes = face_cache["boxes"]
                    affine_matrices = face_cache["affine_matrices"]
                    aligned_faces = face_cache["aligned_faces"]
                    detection_failures = []
                    face_cache_loaded = True
                    print(f"[LatentSync] Loaded face cache: {face_cache_path}")
                else:
                    print(f"[LatentSync] Cache stale, recomputing...")
            except Exception as e:
                print(f"[LatentSync] Cache corrupt ({e}), recomputing...")
                os.remove(face_cache_path)

        if not face_cache_loaded:
            cap = cv2.VideoCapture(video_path)
            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            if len(frames) < 5:
                print(f"[LatentSync] Too few frames ({len(frames)}) in {video_path}")
                return None

            while len(frames) < num_frames:
                frames.append(frames[-1].copy())

            original_frames = np.stack(frames, axis=0)
            boxes = []
            affine_matrices = []
            aligned_faces = []
            detection_failures = []

            # Reset temporal smoothing bias for new video
            image_processor.restorer.p_bias = None

            for i, frame in enumerate(frames):
                try:
                    face, box, affine_matrix = image_processor.affine_transform(frame)
                    boxes.append(box)
                    affine_matrices.append(affine_matrix)
                    aligned_faces.append(face)
                except RuntimeError as e:
                    print(f"[LatentSync] Face detection failed for frame {i}: {e}")
                    boxes.append(None)
                    affine_matrices.append(None)
                    detection_failures.append(i)

            if detection_failures:
                print(f"[LatentSync] Face detection failed for {len(detection_failures)} frames, skipping")
                return None

            os.makedirs(face_detection_cache_dir, exist_ok=True)
            torch.save({
                "aligned_faces": aligned_faces,
                "boxes": boxes,
                "affine_matrices": affine_matrices,
                "resolution": image_processor.resolution,
                "num_frames": len(original_frames),
            }, face_cache_path)
            print(f"[LatentSync] Saved face cache: {face_cache_path}")

        return {
            "video_path": video_path,
            "original_frames": original_frames,
            "num_frames": num_frames,
            "aligned_faces": aligned_faces,
            "boxes": boxes,
            "affine_matrices": affine_matrices,
            "detection_failures": detection_failures if not face_cache_loaded else [],
        }

    except Exception as e:
        print(f"[LatentSync] Preprocessing failed for {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
```

- [ ] **Step 2: Add composite_with_latentsync_float function**

Add directly after `preprocess_with_latentsync`. Copy verbatim from
`inference_v2v.py` lines 264-335:

```python
def composite_with_latentsync_float(generated_float, latentsync_metadata, image_processor,
                                     use_mouth_only_compositing=False):
    """Composite generated faces back onto original video, staying in float space.

    Copied verbatim from OmniAvatar-Train/scripts/inference_v2v.py to ensure
    identical compositing behavior and precision.

    Args:
        generated_float: [T, C, H, W] float tensor in [0, 1]
        latentsync_metadata: dict from preprocess_with_latentsync()
        image_processor: ImageProcessor instance
        use_mouth_only_compositing: blend only mouth region if True

    Returns:
        [T, H, W, 3] uint8 numpy array of composited original-resolution frames
    """
    import torchvision.transforms.functional as TF_v

    original_frames = latentsync_metadata["original_frames"]
    if original_frames is None:
        video_path = latentsync_metadata["video_path"]
        num_frames = latentsync_metadata.get("num_frames", 81)
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
        original_frames = np.stack(frames, axis=0)

    boxes = latentsync_metadata["boxes"]
    affine_matrices = latentsync_metadata["affine_matrices"]
    detection_failures = latentsync_metadata.get("detection_failures", [])
    aligned_faces = latentsync_metadata.get("aligned_faces", None)

    composite_frames = []

    for i in range(generated_float.shape[0]):
        if i in detection_failures or boxes[i] is None:
            composite_frames.append(original_frames[i])
            continue

        face = generated_float[i]  # [C, H, W] float [0,1]

        # Mouth-only compositing in float space (no uint8 quantization)
        if use_mouth_only_compositing and aligned_faces is not None:
            mouth_mask = image_processor.mask_image.float()  # [C, H, W] float32
            original_aligned_float = aligned_faces[i].float() / 255.0  # uint8 -> [0,1]
            face = face * (1 - mouth_mask) + original_aligned_float * mouth_mask

        # Resize in float space
        x1, y1, x2, y2 = boxes[i]
        height = int(y2 - y1)
        width = int(x2 - x1)
        face_resized = TF_v.resize(
            face, size=[height, width],
            interpolation=TF_v.InterpolationMode.BICUBIC, antialias=True,
        )

        # Convert [0,1] -> [-1,1] for restore_img (NO uint8 round-trip)
        face_resized = face_resized * 2.0 - 1.0

        try:
            restored_frame = image_processor.restorer.restore_img(
                original_frames[i], face_resized, affine_matrices[i]
            )
            composite_frames.append(restored_frame)
        except Exception as e:
            print(f"[LatentSync] Restoration failed for frame {i}: {e}")
            composite_frames.append(original_frames[i])

    return np.stack(composite_frames)
```

- [ ] **Step 3: Add save_frames_as_video helper**

Add after `mux_video_with_audio`. Copy from `inference_v2v.py` line 1056-1070:

```python
def save_frames_as_video(frames_np, output_path, fps=25):
    """Save [N, H, W, 3] uint8 numpy array as mp4 video.

    Uses CRF 13 + macro_block_size=None to match LatentSync-train's write_video().
    """
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = imageio.get_writer(
        output_path, fps=fps, codec='libx264',
        macro_block_size=None,
        ffmpeg_params=["-crf", "13"],
        ffmpeg_log_level="error",
    )
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()
```

- [ ] **Step 4: Add load_image_processor helper**

Add after `load_or_encode_text`:

```python
def load_image_processor(mask_path, device):
    """Load LatentSync ImageProcessor for face detection and alignment.

    Lazy import — only called when --latentsync is set.

    Returns:
        ImageProcessor instance with face detector + affine restorer.
    """
    import os as _os
    _os.environ.setdefault("ORT_DISABLE_THREAD_AFFINITY", "1")  # Must be set before insightface import
    from OmniAvatar.utils.latentsync.image_processor import ImageProcessor

    print("Loading LatentSync ImageProcessor ...")
    processor = ImageProcessor(resolution=512, device=device, mask_image=mask_path)
    return processor
```

- [ ] **Step 5: Verify syntax**

Run:
```bash
python -c "import ast; ast.parse(open('scripts/inference/inference_causal.py').read())"
```

Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat: add LatentSync preprocessing, compositing, and helper functions"
```

---

### Task 3: Restructure main() for Batch + LatentSync

**Files:**
- Modify: `scripts/inference/inference_causal.py` (main function)

- [ ] **Step 1: Add enumerate_samples helper**

Add before `main()`:

```python
def enumerate_samples(args):
    """Yield (sample_name, video_path, audio_path, precomputed_dir) for each sample.

    In single-video mode: yields one sample from --video_path.
    In batch mode: scans --input_dir subdirectories.
    """
    if args.input_dir is not None:
        for entry in sorted(os.listdir(args.input_dir)):
            sample_dir = os.path.join(args.input_dir, entry)
            if not os.path.isdir(sample_dir):
                continue
            video_path = os.path.join(sample_dir, "sub_clip.mp4")
            if not os.path.isfile(video_path):
                continue
            audio_path = os.path.join(sample_dir, "audio.wav")
            if not os.path.isfile(audio_path):
                print(f"[Skip] No audio.wav in {sample_dir}")
                continue
            # Check for precomputed tensors
            precomputed = sample_dir if os.path.isfile(
                os.path.join(sample_dir, "vae_latents_mask_all.pt")
            ) else None
            yield entry, video_path, audio_path, precomputed
    else:
        # Single-video mode
        name = os.path.splitext(os.path.basename(args.video_path))[0]
        audio_path = args.audio_path  # May be None (resolve_audio handles extraction)
        yield name, args.video_path, audio_path, args.precomputed_dir
```

- [ ] **Step 2: Replace main() with batch-capable version**

Replace the entire `main()` function with:

```python
def main():
    args = parse_args()
    validate_args(args)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Load models once ---
    print("Loading diffusion model ...")
    model = load_diffusion_model(args, device, dtype)

    print("Loading VAE ...")
    vae = load_vae(args.vae_path, device)

    wav2vec_model, wav2vec_extractor = None, None
    text_embeds_shared = None
    image_processor = None

    # Wav2Vec + T5: load unless all samples will use precomputed tensors
    # (In batch mode we can't know in advance, so load them; they're freed per-sample if unused)
    if not args.precomputed_dir or args.input_dir:
        print("Loading Wav2Vec2 ...")
        wav2vec_model, wav2vec_extractor = load_wav2vec(args.wav2vec_path, device)
        print("Loading text embeddings ...")
        text_embeds_shared = load_or_encode_text(args, device, dtype)

    # LatentSync face processor
    if args.latentsync:
        image_processor = load_image_processor(args.mask_path, device)

    # --- Process samples ---
    total, succeeded, skipped, failed = 0, 0, 0, 0

    for sample_name, video_path, audio_path_sample, precomputed_dir in enumerate_samples(args):
        total += 1

        # Determine output path
        if args.input_dir is not None:
            output_path = os.path.join(args.output_dir, f"{sample_name}.mp4")
        else:
            output_path = args.output_path

        # Skip existing
        if args.skip_existing and os.path.isfile(output_path):
            print(f"[{total}] Skipping {sample_name} (output exists)")
            skipped += 1
            continue

        print(f"\n[{total}] Processing {sample_name} ...")

        try:
            # --- Resolve audio ---
            if audio_path_sample is not None:
                audio_path = audio_path_sample
                tmp_audio = None
            else:
                # Extract from video (single-video mode without --audio_path)
                class _FakeArgs:
                    pass
                _fa = _FakeArgs()
                _fa.video_path = video_path
                _fa.audio_path = args.audio_path
                audio_path, tmp_audio = resolve_audio(_fa)

            try:
                # --- Compute generation length ---
                num_latent_frames, num_video_frames = compute_generation_length(
                    audio_path, args.num_latent_frames, args.chunk_size, args.fps,
                )
                print(f"  {num_latent_frames} latent frames, {num_video_frames} video frames")

                # --- LatentSync preprocessing ---
                latentsync_metadata = None
                if args.latentsync:
                    latentsync_metadata = preprocess_with_latentsync(
                        video_path, image_processor, args.face_cache_dir, num_video_frames,
                    )
                    if latentsync_metadata is None:
                        print(f"  [FAIL] LatentSync preprocessing failed, skipping")
                        failed += 1
                        continue

                # --- Build conditioning ---
                if precomputed_dir is not None:
                    condition = build_condition_from_precomputed(
                        precomputed_dir, args.mask_path,
                        num_latent_frames, device, dtype,
                    )
                else:
                    # Get reference video frames (aligned 512x512 if latentsync)
                    if args.latentsync and latentsync_metadata is not None:
                        # Use aligned faces as 512x512 reference frames
                        aligned_faces = latentsync_metadata["aligned_faces"]
                        # aligned_faces is list of [C, H, W] uint8 tensors → [N, H, W, C] numpy
                        ref_frames_np = np.stack([
                            f.permute(1, 2, 0).numpy() if isinstance(f, torch.Tensor)
                            else f
                            for f in aligned_faces[:num_video_frames]
                        ], axis=0)
                    else:
                        ref_frames_np = load_and_adjust_video(video_path, num_video_frames)

                    text_embeds = text_embeds_shared
                    if text_embeds is None:
                        text_embeds = load_or_encode_text(args, device, dtype)

                    condition = build_condition(
                        vae, wav2vec_model, wav2vec_extractor, ref_frames_np,
                        audio_path, text_embeds, args.mask_path,
                        num_video_frames, num_latent_frames, device, dtype,
                    )

                # --- Run inference ---
                print("  Running AR inference ...")
                output_latents = run_inference(
                    model, condition, num_latent_frames, args.t_list,
                    args.chunk_size, args.context_noise, args.seed, device, dtype,
                )

                # --- Post-processing ---
                if args.latentsync and latentsync_metadata is not None:
                    # VAE decode to float tensor
                    latent_for_vae = output_latents[0].to(torch.float32)
                    video_decoded = vae.decode([latent_for_vae], device=device)
                    video_decoded = video_decoded.clamp(-1, 1)
                    # [1, 3, T_video, H, W] -> [T, 3, H, W] in [0, 1]
                    generated_float = video_decoded[0].permute(1, 0, 2, 3)  # [3, T, H, W] -> [T, 3, H, W]
                    generated_float = (generated_float + 1) / 2  # [-1,1] -> [0,1]

                    # Composite onto original resolution
                    print("  Compositing with LatentSync ...")
                    composited_np = composite_with_latentsync_float(
                        generated_float.cpu(), latentsync_metadata, image_processor,
                        use_mouth_only_compositing=args.use_mouth_only,
                    )

                    # Save composited video + audio
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                    duration_s = composited_np.shape[0] / args.fps
                    tmp_composited = output_path + ".tmp_composited.mp4"
                    save_frames_as_video(composited_np, tmp_composited, fps=args.fps)
                    mux_video_with_audio(tmp_composited, audio_path, output_path, duration_s=duration_s)
                    if os.path.exists(tmp_composited):
                        os.remove(tmp_composited)

                    # Also save aligned 512x512 video + audio (for metrics)
                    aligned_path = output_path.replace(".mp4", "_aligned.mp4")
                    generated_np = (generated_float.permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().cpu().numpy()
                    tmp_aligned = aligned_path + ".tmp.mp4"
                    save_frames_as_video(generated_np, tmp_aligned, fps=args.fps)
                    aligned_duration = generated_np.shape[0] / args.fps
                    mux_video_with_audio(tmp_aligned, audio_path, aligned_path, duration_s=aligned_duration)
                    if os.path.exists(tmp_aligned):
                        os.remove(tmp_aligned)

                    print(f"  Saved: {output_path} (composited) + {aligned_path} (aligned)")
                else:
                    # Standard decode + save (no compositing)
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                    decode_and_save(vae, output_latents, audio_path, output_path, args.fps, device)
                    print(f"  Saved: {output_path}")

                succeeded += 1

            finally:
                if 'tmp_audio' in dir() and tmp_audio is not None and os.path.exists(tmp_audio):
                    os.remove(tmp_audio)

        except Exception as e:
            print(f"  [FAIL] {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        # Clear per-sample state
        model.clear_caches()
        torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Done: {succeeded}/{total} succeeded, {skipped} skipped, {failed} failed")
```

- [ ] **Step 3: Verify syntax**

Run:
```bash
python -c "import ast; ast.parse(open('scripts/inference/inference_causal.py').read())"
python scripts/inference/inference_causal.py --help
```

Expected: No syntax errors, new args visible.

- [ ] **Step 4: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat: restructure main() for batch inference + LatentSync compositing"
```

---

### Task 4: Test Single-Video Backward Compatibility

**Files:**
- No modifications — verification only

- [ ] **Step 1: Test single-video mode still works (no LatentSync)**

```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_OmniAvatar/FastGen

PRETRAINED="../OmniAvatar-Train/pretrained_models"
SAMPLE="/data/karlo-research_715/workspace/kinemaar/datasets/sample_hallo3_latentsync/0010234f331f491ffacc538958094732_shot_001_000"

CUDA_VISIBLE_DEVICES=0 python scripts/inference/inference_causal.py \
    --video_path "$SAMPLE/sub_clip.mp4" \
    --audio_path "$SAMPLE/audio.wav" \
    --output_path /tmp/test_backward_compat.mp4 \
    --ckpt_path "$PRETRAINED/1.3B-causal-step-0002500.pth" \
    --vae_path "$PRETRAINED/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
    --wav2vec_path "$PRETRAINED/wav2vec2-base-960h" \
    --mask_path "../OmniAvatar-Train/OmniAvatar/utils/latentsync/mask.png" \
    --base_model_paths "$PRETRAINED/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    --omniavatar_ckpt_path "$PRETRAINED/OmniAvatar-1.3B/pytorch_model.pt" \
    --precomputed_dir "$SAMPLE" \
    --num_latent_frames 21 \
    --seed 42
```

Expected: Completes successfully, `0 missing, 0 unexpected` keys, output at `/tmp/test_backward_compat.mp4`.

- [ ] **Step 2: Test batch mode with precomputed tensors**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference/inference_causal.py \
    --input_dir /data/karlo-research_715/workspace/kinemaar/datasets/sample_hallo3_latentsync \
    --output_dir /tmp/test_batch_output \
    --ckpt_path "$PRETRAINED/1.3B-causal-step-0002500.pth" \
    --vae_path "$PRETRAINED/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
    --wav2vec_path "$PRETRAINED/wav2vec2-base-960h" \
    --mask_path "../OmniAvatar-Train/OmniAvatar/utils/latentsync/mask.png" \
    --base_model_paths "$PRETRAINED/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    --omniavatar_ckpt_path "$PRETRAINED/OmniAvatar-1.3B/pytorch_model.pt" \
    --num_latent_frames 21 \
    --skip_existing \
    --seed 42
```

Expected: Processes all 3 samples in `sample_hallo3_latentsync/`, outputs in `/tmp/test_batch_output/`. Summary shows `3/3 succeeded`.

- [ ] **Step 3: Test --skip_existing (re-run should skip all)**

Re-run the same batch command.

Expected: `3 skipped` in summary.

- [ ] **Step 4: Fix any issues and commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "fix: batch inference verified end-to-end"
```

---

### Task 5: Test LatentSync Compositing

**Files:**
- No modifications — verification only

Note: This requires `insightface`, `onnxruntime`, and `kornia` packages.
If not installed, install them first:
```bash
pip install insightface onnxruntime kornia
```

- [ ] **Step 1: Test LatentSync mode with single video**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference/inference_causal.py \
    --video_path "$SAMPLE/sub_clip.mp4" \
    --audio_path "$SAMPLE/audio.wav" \
    --output_path /tmp/test_latentsync.mp4 \
    --ckpt_path "$PRETRAINED/1.3B-causal-step-0002500.pth" \
    --vae_path "$PRETRAINED/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
    --wav2vec_path "$PRETRAINED/wav2vec2-base-960h" \
    --mask_path "../OmniAvatar-Train/OmniAvatar/utils/latentsync/mask.png" \
    --base_model_paths "$PRETRAINED/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    --omniavatar_ckpt_path "$PRETRAINED/OmniAvatar-1.3B/pytorch_model.pt" \
    --latentsync \
    --face_cache_dir /tmp/test_face_cache \
    --num_latent_frames 21 \
    --seed 42
```

Expected:
- Face detection runs, cache saved to `/tmp/test_face_cache/{stem}_face_cache.pt`
- Composited video saved to `/tmp/test_latentsync.mp4` (original resolution)
- Aligned video saved to `/tmp/test_latentsync_aligned.mp4` (512x512)
- Both have audio

- [ ] **Step 2: Verify face cache was created and re-used**

Re-run the same command. Should print `[LatentSync] Loaded face cache: ...` instead of recomputing.

- [ ] **Step 3: Test LatentSync batch mode**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference/inference_causal.py \
    --input_dir /data/karlo-research_715/workspace/kinemaar/datasets/sample_hallo3_latentsync \
    --output_dir /tmp/test_latentsync_batch \
    --ckpt_path "$PRETRAINED/1.3B-causal-step-0002500.pth" \
    --vae_path "$PRETRAINED/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
    --wav2vec_path "$PRETRAINED/wav2vec2-base-960h" \
    --mask_path "../OmniAvatar-Train/OmniAvatar/utils/latentsync/mask.png" \
    --base_model_paths "$PRETRAINED/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    --omniavatar_ckpt_path "$PRETRAINED/OmniAvatar-1.3B/pytorch_model.pt" \
    --latentsync \
    --face_cache_dir /tmp/test_face_cache_batch \
    --num_latent_frames 21 \
    --seed 42
```

Expected: Processes all samples with compositing. Each sample produces `{name}.mp4` (composited) + `{name}_aligned.mp4`.

- [ ] **Step 4: Fix any issues and commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "fix: LatentSync compositing verified end-to-end"
```

---

## Verification Checklist

| Check | How | Expected |
|-------|-----|----------|
| Single-video backward compat | Run without --latentsync | Same output as before |
| Batch processes all samples | --input_dir with 3 samples | 3/3 succeeded |
| --skip_existing works | Re-run batch | 3 skipped |
| Face cache created | Check --face_cache_dir | .pt files present |
| Face cache reused | Re-run --latentsync | "Loaded face cache" in logs |
| Composited video is original resolution | Check output dimensions | Matches input video (not 512x512) |
| Aligned video is 512x512 | Check _aligned.mp4 dimensions | 512x512 |
| Audio muxed in both outputs | Play videos | Audio present |
| Face cache compatible with inference_v2v.py | Load cache from our script in inference_v2v.py | Same keys, resolution |
| Error isolation in batch | Corrupt one sample dir | Other samples still process |
