# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import warnings
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import Dict, List, Literal, Optional, Sequence, Union, cast

import numpy as np
import numpy.typing as npt
from pydantic import Field

from qti.aisw.tools.core.modules.api.definitions.common import (
    AISWBaseModel,
    OpPackageIdentifier,
    ProfilingData,
    ProfilingLevel,
    ProfilingOption,
    Target,
)

# TODO: This is a partial duplicate of qti.aisw.tools.core.modules.api.definitions.common.BackendType.
#       Remove once module API refactoring is complete.


class BackendType(str, Enum):
    """
    Enum representing backend types that are supported by a module.
    """

    CPU = "CPU"
    GPU = "GPU"
    HTP = "HTP"
    HTP_MCP = "HTP_MCP"
    AIC = "AIC"
    LPAI = "LPAI"

    @classmethod
    def offline_preparable_backends(cls):
        return [cls.HTP, cls.AIC, cls.HTP_MCP, cls.LPAI]

    @classmethod
    def quantizable_backends(cls):
        return [cls.CPU, cls.HTP, cls.AIC, cls.HTP_MCP, cls.LPAI]

    @classmethod
    def from_id(cls, backend_id: int) -> "BackendType":
        _ID_TO_BACKEND = {3: cls.CPU, 4: cls.GPU, 6: cls.HTP, 11: cls.HTP_MCP, 8: cls.AIC, 12: cls.LPAI}
        return _ID_TO_BACKEND[backend_id]

    @staticmethod
    def is_valid_backend(backend_str: str):
        return any(backend_str == backend for backend in BackendType._member_map_)

    @staticmethod
    def backend_to_id(backend_str: str):
        return BackendType(backend_str).id

    @property
    def id(self):
        _BACKEND_TO_ID = {"CPU": 3, "GPU": 4, "HTP": 6, "HTP_MCP": 11, "AIC": 8, "LPAI": 12}

        return _BACKEND_TO_ID[self._value_]


# --------------------------- Input Types --------------------- #
InputListInput = str | PathLike
NamedTensorMapping = Dict[str, np.ndarray]
ExecutionInputData = Union[
    InputListInput, np.ndarray, NamedTensorMapping, List[np.ndarray], List[NamedTensorMapping]
]


@dataclass
class ExecutionResult:
    """
    The result of executing a Model or CompiledModel object including any generated profiling data.
    This class is iterable and indexable by output name.
    """

    data: Optional[
        Dict[str, npt.NDArray]
        | Sequence[Dict[str, npt.NDArray]]
        | Dict[str, Dict[str, npt.NDArray]]
        | Dict[str, Sequence[Dict[str, npt.NDArray]]]
    ] = None
    """ The output data from the execution. It contains a data field which is a dictionary of output names to
        numpy arrays or a sequence of dictionaries of output names to numpy arrays or a dictionary of
        graph names to dictionary of output names to numpy arrays."""

    profiling_data: Optional[ProfilingData] = None
    """ The profiling data generated during execution."""

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, output_name: str) -> npt.NDArray:
        if self.data is None:
            raise TypeError("Cannot get item from data. Data is None.")

        # Legacy support for multiple inference
        if isinstance(self.data, Sequence):
            raise TypeError("Cannot get item from data of type: Sequence. Use self.data")

        if isinstance(self.data, dict) and all(isinstance(v, dict) for v in self.data.values()):
            raise TypeError("Cannot get item from data of type: Dict[str, Dict]. Use self.data")

        # multiple inference for lora adapters
        if isinstance(self.data, dict) and all(isinstance(v, Sequence) for v in self.data.values()):
            raise TypeError("Cannot get item from data of type: Dict[str, Sequence[Dict]]. Use self.data")

        safe_data = cast(Dict[str, npt.NDArray], self.data)
        return safe_data[output_name]

    def __iter__(self):
        if self.data is None:
            raise TypeError("Cannot iterate through data. Data is None.")

        # Legacy support for multiple inference
        if isinstance(self.data, Sequence):
            warnings.warn(
                " Data is of type sequence. Iterator will return a sequence of dictionaries."
                " Use self.data for key, value pairs."
            )
            return iter(self.data)

        if isinstance(self.data, dict) and all(isinstance(v, dict) for v in self.data.values()):
            warnings.warn(
                " Data is of type dict. Iterator will return a dictionary of dictionaries."
                " Use self.data for key, value pairs."
            )
            return iter(self.data)

        # multiple inference for lora adapters
        if isinstance(self.data, dict) and all(isinstance(v, Sequence) for v in self.data.values()):
            warnings.warn(
                " Data is of type dict with sequences. Iterator will return a dictionary of sequences."
                " Use self.data for key, value pairs."
            )
            return iter(self.data)

        return iter(self.data.items())


class DspArchitecture(str, Enum):
    v66 = "v66"
    v68 = "v68"
    v69 = "v69"
    v73 = "v73"
    v75 = "v75"
    v79 = "v79"
    v81 = "v81"
    v85 = "v85"

    @classmethod
    def list_options(cls):
        """Returns a list of all DSP architecture options"""
        return [option.value for option in cls]


class PerfProfile(str, Enum):
    LOW_BALANCED = "low_balanced"
    BALANCED = "balanced"
    DEFAULT = "default"
    HIGH_PERFORMANCE = "high_performance"
    SUSTAINED_HIGH_PERFORMANCE = "sustained_high_performance"
    BURST = "burst"
    EXTREME_POWER_SAVER = "extreme_power_saver"
    LOW_POWER_SAVER = "low_power_saver"
    POWER_SAVER = "power_saver"
    HIGH_POWER_SAVER = "high_power_saver"
    SYSTEM_SETTINGS = "system_settings"
    NO_USER_INPUT = "no_user_input"
    CUSTOM = "custom"
    INVALID = "invalid"

    @classmethod
    def list_options(cls):
        """Returns a list of all performance profile options"""
        return [option.value for option in cls]


class ContextExecuteConfig(AISWBaseModel):
    """Context priority Configuration for HTP Backend"""

    context_priority: Literal["low", "normal", "normal_high", "high"] = "normal"
    """
    Specifies priority of the context as a context config.
    """

    async_execute_queue_depth: int = 0
    """
    Specifies the number of executions that can be in the queue at a given time.
    """

    enable_graphs: list[str] = Field(default_factory=list)
    """
    Indicates to the backend, during offline prepare, to not load specified graphs into memory.
    """

    memory_limit_hint: int = 0
    """
    Sets the peak memory limit hint of a deserialized context in MBs.
    """

    is_persistent_binary: bool = False
    """
    Indicates that the context binary pointer is available during QnnContext_createFromBinary
    and until QnnContext_free is called.
    """

    cache_compatibility_mode: Literal["permissive", "strict"] = "permissive"
    """
    Specifies the mode used to check whether cache record is optimal for the device.
    The available modes indicate binary cache compatibility:

    - permissive: Binary cache is compatible, if it could run on the device
    - strict: Binary cache is compatible if it could run on the device and fully utilize
            hardware capability. If it cannot fully utilize hardware, selecting this
            option results in a recommendation to prepare the cache again.

    """
