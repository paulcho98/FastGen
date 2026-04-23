#!/usr/bin/env python3
"""LMD-only evaluator (reuses dlib landmark helpers from eval_ssim_lmd.py).

Skips SSIM compute entirely, which is the expensive part of --ssim-lmd
(mediapipe face detect + resize + skimage SSIM per frame). This script
runs dlib 68-landmark detection once per frame pair and computes the
mouth-LMD from the last 20 points (indices 48-67), matching the
paper definition and eval_ssim_lmd.py's `compute_lmd()`.

Output: writes a `ssim_lmd_per_video.log` format compatible with the
downstream aggregator in eval_redmd_taew_audiofix_syncnet_mouthweight.sh:
  mean_lmd: <float>
  <video_name>: <lmd>
  ...

Usage:
  python scripts/eval/eval_lmd_only.py \\
    --real_videos_dir /path/to/gt \\
    --fake_videos_dir /tmp/eval_snm_<LABEL> \\
    --shape_predictor_path /home/work/.local/eval_metrics/shape_predictor_68_face_landmarks.dat \\
    --log_path /path/to/ssim_lmd_per_video.log
"""
import argparse
import sys
from pathlib import Path

# Put the eval_metrics/eval dir on sys.path so we can import eval_ssim_lmd
EVAL_METRICS_DIR = Path("/home/work/.local/eval_metrics/eval")
sys.path.insert(0, str(EVAL_METRICS_DIR))

import cv2  # noqa: E402
import dlib  # noqa: E402
import numpy as np  # noqa: E402
from eval_ssim_lmd import compute_lmd  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_videos_dir", type=Path, required=True)
    p.add_argument("--fake_videos_dir", type=Path, required=True)
    p.add_argument("--shape_predictor_path", type=Path, required=True)
    p.add_argument("--log_path", type=Path, required=True)
    p.add_argument(
        "--name_list_path",
        type=Path,
        default=None,
        help="Optional file with one video_name per line to restrict evaluation.",
    )
    return p.parse_args()


def stem_of(path: Path) -> str:
    # Match the naming convention used by eval_ssim_lmd.py pairing logic:
    # both real and fake are keyed by the stem, minus any _cfr25 suffix on real side.
    s = path.stem
    if s.endswith("_cfr25"):
        s = s[: -len("_cfr25")]
    return s


def compute_video_lmd(real_path: Path, fake_path: Path, detector, predictor):
    cap_real = cv2.VideoCapture(str(real_path))
    cap_fake = cv2.VideoCapture(str(fake_path))
    if not cap_real.isOpened():
        raise RuntimeError(f"Failed to open real video: {real_path}")
    if not cap_fake.isOpened():
        raise RuntimeError(f"Failed to open fake video: {fake_path}")

    real_count = int(cap_real.get(cv2.CAP_PROP_FRAME_COUNT))
    fake_count = int(cap_fake.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(real_count, fake_count) if (real_count > 0 and fake_count > 0) else None

    total_lmd = 0.0
    count = 0
    idx = 0
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        ok_r, fr = cap_real.read()
        ok_f, ff = cap_fake.read()
        if not ok_r or not ok_f:
            break
        if fr.shape != ff.shape:
            ff = cv2.resize(ff, (fr.shape[1], fr.shape[0]))
        lmd = compute_lmd(fr, ff, detector, predictor)
        if lmd is not None:
            total_lmd += lmd
            count += 1
        idx += 1
    cap_real.release()
    cap_fake.release()
    if count == 0:
        return None
    return total_lmd / count


def main():
    args = parse_args()

    name_filter = None
    if args.name_list_path is not None:
        with open(args.name_list_path) as f:
            name_filter = set(line.strip() for line in f if line.strip())

    real_map = {stem_of(p): p for p in args.real_videos_dir.glob("**/*.mp4")}
    fake_map = {stem_of(p): p for p in args.fake_videos_dir.glob("**/*.mp4")}
    common = sorted(set(real_map) & set(fake_map))
    if name_filter is not None:
        common = [n for n in common if n in name_filter]

    if not common:
        raise SystemExit(
            f"No overlapping video stems between {args.real_videos_dir} and {args.fake_videos_dir}"
        )

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(args.shape_predictor_path))

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    per_video_lines = []
    values = []
    for name in common:
        r, f = real_map[name], fake_map[name]
        lmd = compute_video_lmd(r, f, detector, predictor)
        if lmd is None:
            per_video_lines.append(f"{name}: N/A (no frames with detected landmarks)")
            continue
        per_video_lines.append(f"{name}: {lmd:.6f}")
        values.append(lmd)

    mean_lmd = float(np.mean(values)) if values else float("nan")
    with open(args.log_path, "w") as f:
        f.write(f"mean_lmd: {mean_lmd:.6f}\n")
        f.write(f"num_videos_scored: {len(values)} / {len(common)}\n")
        f.write("\n".join(per_video_lines) + "\n")

    print(f"[eval_lmd_only] {len(values)}/{len(common)} videos scored")
    print(f"[eval_lmd_only] mean_lmd = {mean_lmd:.6f}")
    print(f"[eval_lmd_only] wrote {args.log_path}")


if __name__ == "__main__":
    main()
