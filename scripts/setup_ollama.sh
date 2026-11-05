#!/usr/bin/env bash
# Pull the system LLM and create an alias with num_ctx=32768 and temperature=0 baked in.
# Ollama's OpenAI-compatible endpoint may ignore per-request num_ctx, so the alias is what
# chronos actually calls (OLLAMA_SERVED_MODEL).
set -euo pipefail

BASE="${OLLAMA_MODEL:-qwen2.5:7b-instruct-q4_K_M}"
ALIAS="${OLLAMA_SERVED_MODEL:-chronos-qwen2.5-7b-32k}"

command -v ollama >/dev/null || { echo "ollama not installed" >&2; exit 1; }

ollama pull "$BASE"

MODELFILE="$(mktemp)"
trap 'rm -f "$MODELFILE"' EXIT
cat > "$MODELFILE" <<EOF
FROM $BASE
PARAMETER num_ctx 32768
PARAMETER temperature 0
EOF

ollama create "$ALIAS" -f "$MODELFILE"
echo "created $ALIAS from $BASE"
ollama show "$ALIAS" --parameters
