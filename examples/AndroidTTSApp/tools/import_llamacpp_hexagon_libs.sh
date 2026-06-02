#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <llama.cpp pkg-snapdragon lib dir>"
  echo "Example: $0 /path/to/pkg-snapdragon/llama.cpp/lib"
  exit 1
fi

SRC_DIR="$1"
if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Source directory not found: ${SRC_DIR}"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${ROOT_DIR}/app/src/main/jniLibs/arm64-v8a"

mkdir -p "${DEST_DIR}"

required=(
  "libggml-hexagon.so"
  "libggml-htp-v68.so"
  "libggml-htp-v69.so"
  "libggml-htp-v73.so"
  "libggml-htp-v75.so"
  "libggml-htp-v79.so"
  "libggml-htp-v81.so"
)

copied=0
for lib in "${required[@]}"; do
  src="${SRC_DIR}/${lib}"
  if [[ -f "${src}" ]]; then
    cp -f "${src}" "${DEST_DIR}/${lib}"
    echo "Copied ${lib}"
    copied=$((copied + 1))
  else
    echo "Missing (skip): ${lib}"
  fi
done

if [[ ${copied} -eq 0 ]]; then
  echo "No Hexagon runtime library copied. Check source path: ${SRC_DIR}"
  exit 3
fi

echo "Done. Copied ${copied} libraries to ${DEST_DIR}"
