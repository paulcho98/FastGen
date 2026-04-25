# Validation SyncNet Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SyncNet-v2 (sync_c) evaluation to the WandbCallback's validation visualization, logging per-sample and mean sync_c scores to wandb alongside the generated videos.

**Architecture:** Reuse the existing `SyncCScorer` (from the Re-DMD reward path) inside `WandbCallback.on_validation_step_end`. After VAE-decoding the generated latents to pixels, convert to uint8 face frames and score with `SyncCScorer._score_single`. Load raw audio waveforms from the `audio_path` already present in each validation batch. Accumulate per-sample sync_c values and log the mean + per-sample values in `on_validation_end`.

**Tech Stack:** `SyncCScorer` (fastgen/methods/reward/sync_c_scorer.py), WanVideoVAE decode, wandb logging, scipy.io.wavfile for audio loading.

---

### Task 1: Add SyncNet config options to WandbCallback

**Files:**
- Modify: `fastgen/callbacks/wandb.py:315-334` (WandbCallback.__init__)

- [ ] **Step 1: Add syncnet_eval params to WandbCallback.__init__**

Add three new optional parameters to the `WandbCallback.__init__` signature and store them as instance attributes. Also initialize accumulators for sync scores.

```python
# In WandbCallback.__init__, add these parameters after fps:
    syncnet_checkpoint_path: Optional[str] = None,
    syncnet_vshift: int = 15,
    syncnet_audio_sr: int = 16000,
```

And in the body, store them and initialize:

```python
    self.syncnet_checkpoint_path = syncnet_checkpoint_path
    self.syncnet_vshift = syncnet_vshift
    self.syncnet_audio_sr = syncnet_audio_sr
    self._syncnet_scorer = None  # lazily initialized
    self._val_sync_c_scores: list[float] = []
```

- [ ] **Step 2: Verify the file parses**

Run: `python -c "from fastgen.callbacks.wandb import WandbCallback; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add fastgen/callbacks/wandb.py
git commit -m "feat(wandb): add syncnet eval config params to WandbCallback"
```

---

### Task 2: Add lazy SyncCScorer initialization

**Files:**
- Modify: `fastgen/callbacks/wandb.py` (new method on WandbCallback)

- [ ] **Step 1: Add `_get_syncnet_scorer` method**

Add a method that lazily creates and caches a `SyncCScorer` on first use. Place it after `on_app_begin`. This avoids loading the model when sync eval is not configured.

```python
def _get_syncnet_scorer(self):
    """Lazily load SyncCScorer for validation sync_c eval."""
    if self._syncnet_scorer is not None:
        return self._syncnet_scorer
    if not self.syncnet_checkpoint_path:
        return None
    try:
        from fastgen.methods.reward.sync_c_scorer import SyncCScorer
        self._syncnet_scorer = SyncCScorer(
            checkpoint_path=self.syncnet_checkpoint_path,
            input_fps=25.0,
            audio_sample_rate=self.syncnet_audio_sr,
            vshift=self.syncnet_vshift,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float32,
        )
        logger.info(f"[WandbCallback] Loaded SyncCScorer from {self.syncnet_checkpoint_path}")
    except Exception as e:
        logger.warning(f"[WandbCallback] Failed to load SyncCScorer: {e}")
        self.syncnet_checkpoint_path = None  # disable further attempts
    return self._syncnet_scorer
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from fastgen.callbacks.wandb import WandbCallback; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add fastgen/callbacks/wandb.py
git commit -m "feat(wandb): add lazy SyncCScorer init for val sync eval"
```

---

### Task 3: Add audio waveform loading utility

**Files:**
- Modify: `fastgen/callbacks/wandb.py` (new module-level function)

- [ ] **Step 1: Add `_load_audio_waveform` function**

Add a standalone function near the top of the file (after the existing `_to_wandb_with_audio` function) that loads a raw audio waveform from a .wav path, matching the format expected by `SyncCScorer`. This reuses the same loading logic as `_load_raw_waveform_for_reward` in the dataloader but is self-contained.

```python
def _load_audio_waveform(audio_path: str, target_sr: int = 16000, num_frames: int = 81, fps: float = 25.0) -> Optional[torch.Tensor]:
    """Load audio waveform from a .wav file for SyncCScorer evaluation.

    Returns [L] float32 tensor or None on failure.
    """
    if not audio_path or not os.path.isfile(audio_path):
        return None
    try:
        import scipy.io.wavfile as wavfile
        from scipy import signal

        sr, wav = wavfile.read(audio_path)
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        elif wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        wav = torch.from_numpy(wav)
        if wav.ndim == 2:
            wav = wav.mean(dim=1)
        if sr != target_sr:
            num_samples_new = int(len(wav) * target_sr / sr)
            wav = torch.from_numpy(signal.resample(wav.numpy(), num_samples_new))
        target_length = int(num_frames / fps * target_sr)
        if wav.shape[0] < target_length:
            wav = torch.nn.functional.pad(wav, (0, target_length - wav.shape[0]))
        else:
            wav = wav[:target_length]
        return wav.to(torch.float32)
    except Exception as e:
        logger.warning(f"[SyncEval] Failed to load audio from {audio_path}: {e}")
        return None
```

Also add `import numpy as np` at the top of the file if not already present (it isn't — check and add).

- [ ] **Step 2: Verify the function loads**

Run: `python -c "from fastgen.callbacks.wandb import _load_audio_waveform; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add fastgen/callbacks/wandb.py
git commit -m "feat(wandb): add audio waveform loading for sync eval"
```

---

### Task 4: Integrate SyncNet scoring into `on_validation_step_end`

**Files:**
- Modify: `fastgen/callbacks/wandb.py:613-664` (`on_validation_step_end` method)

This is the core integration. After the existing VAE decode block (which produces `gen_decoded` on rank 0), add SyncNet scoring.

- [ ] **Step 1: Add sync_c scoring after VAE decode in `on_validation_step_end`**

Inside the `if _rank == 0:` block, after the line `self._val_audio_paths.append(audio_path)` (around line 660), add:

```python
                # SyncNet-v2 evaluation on generated video
                scorer = self._get_syncnet_scorer()
                if scorer is not None and audio_path is not None:
                    try:
                        audio_wav = _load_audio_waveform(
                            audio_path, target_sr=self.syncnet_audio_sr,
                        )
                        if audio_wav is not None:
                            # gen_decoded is [1, C, T_pix, H, W] float in [-1, 1]
                            # Convert to [T_pix, 3, H, W] uint8
                            pixels = gen_decoded.clamp(-1.0, 1.0)
                            u8 = ((pixels + 1.0) * 127.5).to(torch.uint8)
                            face_frames = u8[0].permute(1, 0, 2, 3).contiguous()  # [T, 3, H, W]
                            sync_c = scorer._score_single(face_frames, audio_wav)
                            self._val_sync_c_scores.append(sync_c.item())
                            logger.info(
                                f"[SyncEval] val sample {step}: sync_c={sync_c.item():.3f}"
                            )
                    except Exception as e:
                        logger.warning(f"[SyncEval] Failed on val sample {step}: {e}")
```

- [ ] **Step 2: Verify no syntax errors**

Run: `python -c "from fastgen.callbacks.wandb import WandbCallback; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add fastgen/callbacks/wandb.py
git commit -m "feat(wandb): score generated val videos with SyncCScorer"
```

---

### Task 5: Log sync_c metrics in `on_validation_end`

**Files:**
- Modify: `fastgen/callbacks/wandb.py:666-687` (`on_validation_end` method)

- [ ] **Step 1: Add sync_c logging to `on_validation_end`**

At the end of the method, before the final list-clearing block, add sync_c wandb logging. Also include per-sample scores in the video captions.

First, modify the video logging block (around line 671-682) to include sync_c in captions. Replace the existing gen_list/gt_list construction loop with one that includes sync_c:

```python
            for i, (gen_v, gt_v) in enumerate(zip(self._val_gen_videos, self._val_gt_videos)):
                ap = self._val_audio_paths[i] if i < len(self._val_audio_paths) else None
                # Build caption with sync_c if available
                caption = None
                if i < len(self._val_sync_c_scores):
                    caption = f"sync_c={self._val_sync_c_scores[i]:.3f}"
                if ap:
                    gen_list.append(tensor_to_wandb_video_with_audio(gen_v, ap, fps=self.fps, caption=caption))
                    gt_list.append(tensor_to_wandb_video_with_audio(gt_v, ap, fps=self.fps))
                else:
                    gen_list.append(wandb.Video(gen_v[0].numpy(), fps=self.fps, format="mp4", caption=caption))
                    gt_list.append(wandb.Video(gt_v[0].numpy(), fps=self.fps, format="mp4"))
```

Then add sync_c scalar logging right after the video log block (after the `wandb.log` for videos):

```python
            if self._val_sync_c_scores:
                mean_sync_c = sum(self._val_sync_c_scores) / len(self._val_sync_c_scores)
                wandb.log({f"val{idx}/sync_c_mean": mean_sync_c}, step=iteration)
                for i, sc in enumerate(self._val_sync_c_scores):
                    wandb.log({f"val{idx}/sync_c_sample_{i}": sc}, step=iteration)
                logger.info(
                    f"[SyncEval] val{idx} mean sync_c: {mean_sync_c:.3f} "
                    f"(n={len(self._val_sync_c_scores)})"
                )
```

Finally, add `self._val_sync_c_scores = []` to the clearing block at the end (alongside the existing list clears):

```python
        self._val_gen_videos = []
        self._val_gt_videos = []
        self._val_audio_paths = []
        self._val_sync_c_scores = []
```

- [ ] **Step 2: Verify no syntax errors**

Run: `python -c "from fastgen.callbacks.wandb import WandbCallback; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add fastgen/callbacks/wandb.py
git commit -m "feat(wandb): log sync_c metrics in validation end"
```

---

### Task 6: Wire up syncnet config in experiment config

**Files:**
- Modify: `fastgen/configs/experiments/OmniAvatar/config_sf.py:196-205` (WandbCallback config section)

- [ ] **Step 1: Add syncnet_checkpoint_path to the wandb callback config**

In `config_sf.py`, after the existing `config.trainer.callbacks.wandb.fps = 25` line (around line 201), add:

```python
    config.trainer.callbacks.wandb.syncnet_checkpoint_path = "/home/work/.local/eval_metrics/checkpoints/auxiliary/syncnet_v2.model"
```

- [ ] **Step 2: Verify config creates successfully**

Run:
```bash
cd /home/work/.local/hyunbin/FastGen-redmd && python -c "
from fastgen.configs.experiments.OmniAvatar.config_sf import create_config
c = create_config()
print('syncnet_checkpoint_path:', c.trainer.callbacks.wandb.syncnet_checkpoint_path)
print('OK')
"
```
Expected: prints the path and `OK`

- [ ] **Step 3: Commit**

```bash
git add fastgen/configs/experiments/OmniAvatar/config_sf.py
git commit -m "feat(config): enable syncnet val eval in SF experiment config"
```

---

### Task 7: Smoke test the full validation pipeline

**Files:** None (verification only)

- [ ] **Step 1: Verify SyncCScorer loads and scores a dummy input**

Run:
```bash
cd /home/work/.local/hyunbin/FastGen-redmd && python -c "
from fastgen.methods.reward.sync_c_scorer import SyncCScorer
import torch
scorer = SyncCScorer(
    checkpoint_path='/home/work/.local/eval_metrics/checkpoints/auxiliary/syncnet_v2.model',
    device='cuda', dtype=torch.float32,
)
# Dummy face-aligned frames: 81 frames, 3ch, 512x512, uint8
video = torch.randint(0, 255, (81, 3, 512, 512), dtype=torch.uint8)
audio = torch.randn(51200)  # 3.2 seconds at 16kHz
score = scorer._score_single(video, audio)
print(f'sync_c = {score.item():.4f}')
print('Scorer smoke test PASSED')
"
```
Expected: prints a sync_c value and `Scorer smoke test PASSED`

- [ ] **Step 2: Verify audio loading utility**

Run:
```bash
cd /home/work/.local/hyunbin/FastGen-redmd && python -c "
from fastgen.callbacks.wandb import _load_audio_waveform
import os, glob
# Find a real audio file from training data
audio_files = glob.glob('/home/work/stableavatar_data/v2v_training_data/*/audio.wav')
if audio_files:
    wav = _load_audio_waveform(audio_files[0])
    print(f'Loaded waveform: shape={wav.shape}, dtype={wav.dtype}')
    print('Audio loading PASSED')
else:
    print('No audio files found — skipping test')
"
```
Expected: prints waveform shape and `Audio loading PASSED`

- [ ] **Step 3: Verify WandbCallback instantiation with syncnet params**

Run:
```bash
cd /home/work/.local/hyunbin/FastGen-redmd && python -c "
from fastgen.callbacks.wandb import WandbCallback
cb = WandbCallback(
    fps=25,
    syncnet_checkpoint_path='/home/work/.local/eval_metrics/checkpoints/auxiliary/syncnet_v2.model',
)
print(f'syncnet_checkpoint_path: {cb.syncnet_checkpoint_path}')
scorer = cb._get_syncnet_scorer()
print(f'Scorer loaded: {scorer is not None}')
print('Callback instantiation PASSED')
"
```
Expected: prints path, `True`, and `Callback instantiation PASSED`

- [ ] **Step 4: Commit (no changes — verification only)**

No commit needed. All tasks complete.
