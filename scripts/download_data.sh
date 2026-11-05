#!/usr/bin/env bash
# Download LoCoMo and LongMemEval-S from their official sources at pinned revisions and record
# sha256 checksums in data/raw/CHECKSUMS. If CHECKSUMS already lists a file, the download is
# verified against it (so a changed upstream file fails loudly instead of silently drifting).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
SUMS="$RAW/CHECKSUMS"
mkdir -p "$RAW"

# Official sources, pinned.
LOCOMO_REV="3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"   # snap-research/locomo main @ 2024-08-13
LME_REV="2ec2a557f339b6c0369619b1ed5793734cc87533"      # HF xiaowu0162/longmemeval @ 2025-09-19

LOCOMO_URL="https://raw.githubusercontent.com/snap-research/locomo/${LOCOMO_REV}/data/locomo10.json"
# Upstream file has no extension; we store it as longmemeval_s.json.
LME_URL="https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/${LME_REV}/longmemeval_s"

sha() { shasum -a 256 "$1" | awk '{print $1}'; }

fetch() {
  local url="$1" name="$2" dest="$RAW/$2"
  if [[ ! -s "$dest" ]]; then
    echo "downloading $name"
    curl -fL --retry 3 --progress-bar -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
  else
    echo "have $name"
  fi
  local got expected
  got="$(sha "$dest")"
  expected="$(awk -v n="$name" '$2 == n {print $1}' "$SUMS" 2>/dev/null || true)"
  if [[ -n "$expected" ]]; then
    if [[ "$got" != "$expected" ]]; then
      echo "CHECKSUM MISMATCH for $name: expected $expected got $got" >&2
      exit 1
    fi
    echo "  sha256 ok"
  else
    echo "$got  $name" >> "$SUMS"
    echo "  recorded sha256 $got"
  fi
}

fetch "$LOCOMO_URL" "locomo10.json"
fetch "$LME_URL" "longmemeval_s.json"
