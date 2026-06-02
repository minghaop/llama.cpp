# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import sys
from pathlib import Path

modeltools = None


try:
    from qti.aisw.converters.common import modeltools as qairt_tools_cpp
except ImportError:
    try:
        from qti.aisw.dlc_utils import modeltools as qairt_tools_cpp
    except ImportError:
        print("ERROR: Unable to import qti.aisw.converters.common.modeltools")
        sys.exit(1)


try:
    from qti.aisw.tools.core import modules as qti_modules
    from qti.aisw.tools.core.modules import api as qti_module_api
    from qti.aisw.tools.core.modules import context_bin_gen
except ImportError:
    print("ERROR: Unable to import qti.aisw.tools.core.modules")
    sys.exit(1)
