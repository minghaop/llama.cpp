#!/usr/bin/env bash
set -euo pipefail

# LiteRT Qualcomm NPU AOT workflow wrapper.
#
# Host requirements from official docs:
#   1) Ubuntu 22.04 LTS (recommended)
#   2) Linux x86_64
#
# Usage:
#   ./tools/run_litert_aot_qualcomm.sh \
#     --model app/src/main/assets/models/flow.tflite \
#     --out-dir npu_aot_out_flow_qnn \
#     --model-name flow \
#     --soc SM8750

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: Qualcomm AOT compile must run on Linux x86_64 host. Current OS: $(uname -s)" >&2
  exit 2
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "ERROR: Qualcomm AOT compile must run on x86_64 host. Current arch: $(uname -m)" >&2
  exit 2
fi

if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
    echo "WARN: Official Qualcomm guide recommends Ubuntu 22.04. Current: ${PRETTY_NAME:-unknown}" >&2
  fi
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
  ai-edge-litert-sdk-google-tensor-nightly \
  ai-edge-litert-sdk-qualcomm \
  ai-edge-litert-sdk-qualcomm-nightly || true

if [[ "${LITERT_USE_NIGHTLY}" == "1" || "${LITERT_USE_NIGHTLY}" == "true" || "${LITERT_USE_NIGHTLY}" == "TRUE" ]]; then
  if [[ -n "${LITERT_NIGHTLY_VERSION}" ]]; then
    echo "INFO: installing pinned nightly version: ${LITERT_NIGHTLY_VERSION}"
    python3 -m pip install --pre -U -i "${LITERT_PIP_INDEX_URL}" \
      "ai-edge-litert-nightly==${LITERT_NIGHTLY_VERSION}" \
      "ai-edge-litert-sdk-qualcomm-nightly==${LITERT_NIGHTLY_VERSION}"
  else
    echo "INFO: installing latest nightly versions"
    python3 -m pip install --pre -U -i "${LITERT_PIP_INDEX_URL}" \
      ai-edge-litert-nightly \
      ai-edge-litert-sdk-qualcomm-nightly
  fi
else
  echo "INFO: installing stable versions: ${LITERT_STABLE_VERSION}"
  python3 -m pip install -U -i "${LITERT_PIP_INDEX_URL}" \
    "ai-edge-litert==${LITERT_STABLE_VERSION}" \
    "ai-edge-litert-sdk-qualcomm==${LITERT_STABLE_VERSION}"
fi

python3 "$ROOT_DIR/tools/litert_aot_compile_qualcomm.py" "$@"

cat <<'EOF'

Next steps:
1) Copy generated model(s) into app/src/main/assets/models/ or app files/models/.
2) Rebuild Android app and check LiteRT runtime logs for Qualcomm NPU offloading.
EOF
