import sys

from qairt.optimizer.utils.logger import logger

if sys.platform.startswith("win"):
    # pypi dosen't offer islpy prebuilt package for windows platform
    # if user already installed islpy manually, we can use the isl implementation
    # otherwise, we use the experimental implementation that dosen't require islpy
    has_isl = False
    try:
        import islpy

        has_isl = True
    except ImportError as e:
        logger.warning(
            "Windows platform detected and islpy package is not installed,"
            "Falling back to experimental implementation of reshape-slice-analysis."
        )
        from qairt.optimizer.onnx.passes.mha2sha.reorder_reshape_slice_utils_v2 import (
            get_reshape_slice_reordered_slice_attrs,
        )

    if has_isl:
        from qairt.optimizer.onnx.passes.mha2sha.reorder_reshape_slice_utils_v1 import (
            get_reshape_slice_reordered_slice_attrs,
        )

else:
    from qairt.optimizer.onnx.passes.mha2sha.reorder_reshape_slice_utils_v1 import (
        get_reshape_slice_reordered_slice_attrs,
    )

__all__ = [
    "get_reshape_slice_reordered_slice_attrs",
]
