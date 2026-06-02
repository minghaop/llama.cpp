#!/bin/bash
#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

# This script sets up the various environment variables needed to run sdk binaries and scripts
OPTIND=1

_usage()
{
cat << EOF
Usage: source $(basename "${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}") [-h]

Script sets up environment variables needed for running sdk binaries and scripts

EOF
}

function _setup_qairt_sdk()
{
  # get directory of this bash script
  local SOURCEDIR; SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

  export QAIRT_SDK_ROOT=$(readlink -f "${SOURCEDIR}"/..)
  echo "[INFO] QAIRT_SDK_ROOT=${QAIRT_SDK_ROOT}"

  if [ -d "${QAIRT_SDK_ROOT}/include/QNN" ]; then
    export QNN_SDK_ROOT="$( cd "${QAIRT_SDK_ROOT}" && pwd )"
    export PYTHONPATH="${QNN_SDK_ROOT}/benchmarks/QNN/":${PYTHONPATH}
    if ls "${QNN_SDK_ROOT}"/bin/x86_64-linux-clang/hexagon-* > /dev/null 2>&1; then
        export HEXAGON_TOOLS_DIR=${QNN_SDK_ROOT}/bin/x86_64-linux-clang
    fi
  else
    echo "[WARNING] QNN API files are missing from QAIRT SDK. Please verify the integrity of your download."
  fi

  if [ -d "${QAIRT_SDK_ROOT}/include/SNPE" ]; then
    export SNPE_ROOT="$( cd "${QAIRT_SDK_ROOT}" && pwd )"
  fi

  if [ -z "${PYTHONPATH}" ]; then
    export PYTHONPATH="${QAIRT_SDK_ROOT}/lib/python/"
  else
    export PYTHONPATH="${QAIRT_SDK_ROOT}/lib/python/":${PYTHONPATH}
  fi

  export PATH=${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang:${PATH}

  if [ -z "${LD_LIBRARY_PATH}" ]; then
    export LD_LIBRARY_PATH=${QAIRT_SDK_ROOT}/lib/x86_64-linux-clang
  else
    export LD_LIBRARY_PATH=${QAIRT_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}
  fi
}

function _cleanup()
{
  unset -f _usage
  unset -f _setup_qairt_sdk
  unset -f _cleanup
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  echo "[ERROR] This script should be run with 'source'"
  _usage;
  exit 1;
fi

# parse arguments
while getopts "h?" opt; do
  case ${opt} in
    h  ) _usage; return 0 ;;
    \? ) echo "See -h for help."; return 1 ;;
  esac
done

_setup_qairt_sdk

# cleanup
_cleanup

if [ "x${QNN_SDK_ROOT}" != "x" ] && [ "x${SNPE_ROOT}" != "x" ]; then
  echo "[WARN] QNN_SDK_ROOT/SNPE_ROOT set to QAIRT_SDK_ROOT for backwards compatibility and will be deprecated in a future release."
elif [ "x${QNN_SDK_ROOT}" != "x" ]; then
  echo "[WARN] QNN_SDK_ROOT set to QAIRT_SDK_ROOT for backwards compatibility and will be deprecated in a future release."
fi

echo "[INFO] QAIRT SDK environment setup complete"
