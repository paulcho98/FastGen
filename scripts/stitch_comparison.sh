#!/bin/bash
# Stitch a side-by-side comparison video per clip:
#   row 1: GT | Wan+Wan | Wan+TAEHV
#   row 2: TAEHV+TAEHV | 4step_full | 4step_sw7
# Each panel labelled in the top-left. Audio pulled from GT.
set -euo pipefail

HDTF=/home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr
ROOT=/home/work/.local/hyunbin/FastGen-redmd/modal_out
OUT=$ROOT/stitched_81
mkdir -p "$OUT"

# Panel label metadata: directory, clip-suffix ("" if no suffix), label.
DIRS=(
  "$HDTF"                              ""      "GT"
  "$ROOT/timing_0000600_wan"           ""      "Wan+Wan 2step sw7"
  "$ROOT/timing_0000600_taehv"         ""      "Wan+TAEHV 2step sw7"
  "$ROOT/timing_0000600_taehv_full"    ""      "TAEHV+TAEHV 2step sw7"
  "$ROOT/timing_0003000_4step_full_wan" "" "Wan+Wan 4step full-attn"
  "$ROOT/timing_0003000_4step_sw7_wan"  "" "Wan+Wan 4step sw7"
)

total=$(ls "$HDTF"/*_cfr25.mp4 | wc -l)
i=0

for gt in "$HDTF"/*_cfr25.mp4; do
  name=$(basename "$gt" _cfr25.mp4)
  i=$((i+1))
  out="$OUT/${name}.mp4"
  if [ -f "$out" ]; then
    echo "[$i/$total] $name (skip, exists)"
    continue
  fi
  echo "[$i/$total] $name"

  # Build input paths.
  p_gt="$gt"
  p_wan="$ROOT/timing_0000600_wan/${name}.mp4"
  p_tae="$ROOT/timing_0000600_taehv/${name}.mp4"
  p_ttt="$ROOT/timing_0000600_taehv_full/${name}.mp4"
  p_4f="$ROOT/timing_0003000_4step_full_wan/${name}.mp4"
  p_4s="$ROOT/timing_0003000_4step_sw7_wan/${name}.mp4"

  for p in "$p_wan" "$p_tae" "$p_ttt" "$p_4f" "$p_4s"; do
    if [ ! -f "$p" ]; then
      echo "  skip: missing $p"
      continue 2
    fi
  done

  # Common drawtext options. Use fontfile path that typically exists.
  FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
  if [ ! -f "$FONT" ]; then
    FONT=$(fc-match -f "%{file}" "Sans:bold" 2>/dev/null || echo "")
  fi
  FONTOPT=""
  if [ -n "$FONT" ] && [ -f "$FONT" ]; then
    FONTOPT="fontfile=${FONT}:"
  fi
  TXT_STYLE="${FONTOPT}fontsize=22:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=5:x=10:y=10"

  ffmpeg -y -loglevel error \
    -i "$p_gt" -i "$p_wan" -i "$p_tae" -i "$p_ttt" -i "$p_4f" -i "$p_4s" \
    -filter_complex "\
      [0:v]drawtext=text='GT':${TXT_STYLE}[v0];\
      [1:v]drawtext=text='Wan+Wan 2step sw7':${TXT_STYLE}[v1];\
      [2:v]drawtext=text='Wan+TAEHV 2step sw7':${TXT_STYLE}[v2];\
      [3:v]drawtext=text='TAEHV+TAEHV 2step sw7':${TXT_STYLE}[v3];\
      [4:v]drawtext=text='Wan+Wan 4step full':${TXT_STYLE}[v4];\
      [5:v]drawtext=text='Wan+Wan 4step sw7':${TXT_STYLE}[v5];\
      [v0][v1][v2]hstack=inputs=3[row1];\
      [v3][v4][v5]hstack=inputs=3[row2];\
      [row1][row2]vstack=inputs=2[grid]" \
    -map "[grid]" -map 0:a? \
    -c:v libx264 -crf 20 -pix_fmt yuv420p -preset fast \
    -c:a aac -shortest "$out"
done

echo "Done. Stitched videos in $OUT"
