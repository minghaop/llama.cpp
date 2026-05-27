#!/usr/bin/env bash
set -euo pipefail

# Official LiteRT Google Tensor NPU AOT workflow wrapper.
# Host requirements:
#   1) Linux x86_64
#   2) CPU supports AVX
#
# Usage:
#   GOOGLE_TENSOR_SDK_BETA=/path/to/litert_plugin_compiler.tar.gz \
#   ./tools/run_litert_aot_google_tensor.sh \
#     --model app/src/main/assets/models/flow.tflite \
#     --out-dir npu_aot_out_flow \
#     --soc G5

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: AOT compile must run on Linux x86_64 host. Current OS: $(uname -s)" >&2
  exit 2
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "ERROR: AOT compile must run on x86_64 host. Current arch: $(uname -m)" >&2
  exit 2
fi

if ! grep -qi avx /proc/cpuinfo; then
  echo "ERROR: CPU AVX not detected; Google Tensor compiler plugin requires AVX." >&2
  exit 2
fi

if [[ -z "${GOOGLE_TENSOR_SDK_BETA:-}" ]]; then
  echo "ERROR: GOOGLE_TENSOR_SDK_BETA is not set." >&2
  exit 2
fi

if [[ ! -f "$GOOGLE_TENSOR_SDK_BETA" ]]; then
  echo "ERROR: GOOGLE_TENSOR_SDK_BETA file not found: $GOOGLE_TENSOR_SDK_BETA" >&2
  exit 2
fi

# Optional knobs:
#   LITERT_PIP_INDEX_URL      default: https://pypi.org/simple
#   LITERT_USE_NIGHTLY        default: 1 (1/true=yes, 0/false=no)
#   LITERT_NIGHTLY_VERSION    e.g. 2.2.0.dev20260525 (empty => latest pre-release)
#   LITERT_STABLE_VERSION     default: 2.1.5
LITERT_PIP_INDEX_URL="${LITERT_PIP_INDEX_URL:-https://pypi.org/simple}"
LITERT_USE_NIGHTLY="${LITERT_USE_NIGHTLY:-1}"
LITERT_NIGHTLY_VERSION="${LITERT_NIGHTLY_VERSION:-}"
LITERT_STABLE_VERSION="${LITERT_STABLE_VERSION:-2.1.5}"

echo "INFO: using pip index: ${LITERT_PIP_INDEX_URL}"
python3 -m pip install -U -i "${LITERT_PIP_INDEX_URL}" pip setuptools wheel

# Remove conflicting combos first to avoid mixed stable/nightly imports.
python3 -m pip uninstall -y \
  ai-edge-litert \
  ai-edge-litert-nightly \
  ai-edge-litert-sdk-google-tensor \
  ai-edge-litert-sdk-google-tensor-nightly || true

if [[ "${LITERT_USE_NIGHTLY}" == "1" || "${LITERT_USE_NIGHTLY}" == "true" || "${LITERT_USE_NIGHTLY}" == "TRUE" ]]; then
  if [[ -n "${LITERT_NIGHTLY_VERSION}" ]]; then
    echo "INFO: installing pinned nightly version: ${LITERT_NIGHTLY_VERSION}"
    python3 -m pip install --pre -U -i "${LITERT_PIP_INDEX_URL}" \
      "ai-edge-litert-nightly==${LITERT_NIGHTLY_VERSION}" \
      "ai-edge-litert-sdk-google-tensor-nightly==${LITERT_NIGHTLY_VERSION}"
  else
    echo "INFO: installing latest nightly versions"
    python3 -m pip install --pre -U -i "${LITERT_PIP_INDEX_URL}" \
      ai-edge-litert-nightly \
      ai-edge-litert-sdk-google-tensor-nightly
  fi
else
  echo "INFO: installing stable versions: ${LITERT_STABLE_VERSION}"
  python3 -m pip install -U -i "${LITERT_PIP_INDEX_URL}" \
    "ai-edge-litert==${LITERT_STABLE_VERSION}" \
    "ai-edge-litert-sdk-google-tensor==${LITERT_STABLE_VERSION}"
fi

python3 "$ROOT_DIR/tools/litert_aot_compile_google_tensor.py" "$@"

cat <<'EOF'

Next steps:
1) Copy generated model(s) into app/src/main/assets/models/ or app files/models/.
   Expected names include:
   - flow_Google_Tensor_G5.tflite
   - flow_fallback.tflite
2) Rebuild Android app; runtime will prefer compiled Google Tensor model automatically.
EOF
