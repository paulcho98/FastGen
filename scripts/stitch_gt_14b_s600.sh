#!/bin/bash
# Stitch 3-panel side-by-side comparison: GT | 14B teacher (step-10500) | Re-DMD s600.
# Each clip: 1x3 horizontal stack, labeled, with GT's audio.
# Usage:
#   bash scripts/stitch_gt_14b_s600.sh
set -uo pipefail

GT_DIR=/home/work/.local/OmniAvatar/demo_out/comprehensive_eval/originals/hdtf
T14B_DIR=/home/work/.local/OmniAvatar/demo_out/v2v_eval_phase2_14B/step-10500/hdtf_composited
S600_DIR=/home/work/output_hdtf_sf_redmd_beta2_taew/step_0000600
OUT_DIR=/home/work/output_hdtf_sf_redmd_beta2_taew/stitched_gt_14b_s600
mkdir -p "$OUT_DIR"

FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if [ ! -f "$FONT" ]; then
    FONT=$(fc-match -f "%{file}" "Sans:bold" 2>/dev/null || echo "")
fi
FONTOPT=""
[ -n "$FONT" ] && [ -f "$FONT" ] && FONTOPT="fontfile=${FONT}:"
TXT="${FONTOPT}fontsize=26:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6:x=10:y=10"

total=$(ls "$GT_DIR"/*_cfr25.mp4 | wc -l)
i=0
for gt in "$GT_DIR"/*_cfr25.mp4; do
    name=$(basename "$gt" _cfr25.mp4)
    i=$((i+1))
    t14b="$T14B_DIR/${name}_cfr25.mp4"
    s600="$S600_DIR/${name}.mp4"
    out="$OUT_DIR/${name}.mp4"
    if [ -f "$out" ]; then
        echo "[$i/$total] $name (skip, exists)"
        continue
    fi
    if [ ! -f "$t14b" ]; then echo "[$i/$total] $name — missing 14B, skip"; continue; fi
    if [ ! -f "$s600" ]; then echo "[$i/$total] $name — missing s600, skip"; continue; fi
    echo "[$i/$total] stitching $name"
    ffmpeg -y -loglevel error \
      -i "$gt" -i "$t14b" -i "$s600" \
      -filter_complex "\
        [0:v]drawtext=text='GT':${TXT}[v0];\
        [1:v]drawtext=text='14B teacher (step-10500)':${TXT}[v1];\
        [2:v]drawtext=text='Re-DMD s600':${TXT}[v2];\
        [v0][v1][v2]hstack=inputs=3[grid]" \
      -map "[grid]" -map 0:a? \
      -c:v libx264 -crf 20 -pix_fmt yuv420p -preset fast \
      -c:a aac -shortest "$out"
done
echo ""
echo "Done. Stitched videos in $OUT_DIR ($(ls $OUT_DIR/*.mp4 2>/dev/null | wc -l)/33)"
