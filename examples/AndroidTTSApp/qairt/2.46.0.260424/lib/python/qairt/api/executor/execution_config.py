# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
ExecutionConfig class.
"""

from typing import Dict, List, Literal, Optional, Union

from pydantic import field_validator, model_validator

from qairt.api.common.backends.aic import AicRuntimeConfig
from qairt.api.common.backends.htp import (
    HtpConfigHelper,
    HtpContextConfig,
    HtpDeviceConfig,
    HtpDeviceCoreConfig,
    HtpGraphConfig,
    HtpGroupContextConfig,
    HtpMemoryConfig,
)
from qairt.api.common.backends.htp_mcp import (
    HtpMcpConfigHelper,
    HtpMcpContextConfig,
    HtpMcpCrcConfig,
    HtpMcpDeviceConfig,
    HtpMcpGraphConfig,
)
from qairt.api.configs import DevicePlatformType
from qairt.api.configs.common import AISWBaseModel, BackendType, ContextExecuteConfig, PerfProfile
from qairt.utils.loggers import get_logger
from qti.aisw.tools.core.modules.api import OpPackageIdentifier, ProfilingLevel, ProfilingOption

_execute_config_logger = get_logger("qairt.execute")

supported_initialize_execute_platforms = {
    DevicePlatformType.X86_64_LINUX,
    DevicePlatformType.X86_64_WINDOWS_MSVC,
    DevicePlatformType.WOS,
    DevicePlatformType.LINUX_EMBEDDED,
    DevicePlatformType.ANDROID,
}


class ExecutionConfig(AISWBaseModel):
    """
    Pydantic class of parameters for model execution
    """

    backend: Optional[BackendType | str] = None
    """
    Backend to be used for execution.
    """

    debug: Optional[bool] = None
    """
    Specifies that output from all layers of the network will be saved.
    """

    context_custom_configs: Optional[List[HtpContextConfig] | List[HtpMcpContextConfig]] = None
    """ Context configuration options specific to a backend.
        Only HTP and HTP MCP backend options are supported.
        See `qairt.api.common.backends.htp.config.HtpContextConfig` and
        `qairt.api.common.backends.htp_mcp.config.HtpMcpContextConfig` for options
    """

    device_custom_configs: Optional[List[HtpDeviceConfig] | List[HtpMcpDeviceConfig]] = None
    """
    Device configuration options specific to a backend.
    Only HTP and HTP MCP backend options are supported.
    See `qairt.api.common.backends.htp.config.HtpDeviceConfig`
    and `qairt.api.common.backends.htp.config.HtpMcpDeviceConfig` for options.
    """

    graph_custom_configs: Optional[List[HtpGraphConfig]] = None
    """
    Graph configuration options specific to a backend.
    Currently only used for HTP backend. See :class:`qairt.api.common.backends.htp.config.HtpGraphConfig`
    for options.
    """

    memory_custom_config: Optional[HtpMemoryConfig] = None
    """
    Memory backend configuration for the compiler. Only HTP backend configurations are supported.
    """

    context_execute_custom_config: Optional[ContextExecuteConfig] = None
    """
    Context configuration options for setting the priority, peak memory of a context.
    See `qairt.api.configs.common.ContextExecuteConfig` for options
    """

    runtime_custom_config: Optional[AicRuntimeConfig] = None
    """
    Set this field to enable configurations that are passed by the backend to the executor.
    Note this option is currently only applicable to the AIC Backend.
    Use qairt.api.common.backends.aic.AicRuntimeConfig.list_config_options to see valid fields.
    """

    use_native_output_data: Optional[bool] = None
    """
    Specifies that the output files will be generated
    in the data type native to the graph or in floating point.
    """

    use_native_input_data: Optional[bool] = None
    """
    Specifies that the input files will be parsed in the data type
    native to the graph. If not specified, input files will be parsed in floating point.
    Note that options use_native_input_data and native_input_tensor_names are mutually exclusive.
    """

    native_input_tensor_names: Optional[List[str]] = None
    """
    List of input tensor names,for which the input files
    would be read/parsed in native format. Note that
    options use_native_input_data and native_input_tensor_names are mutually exclusive.
    """

    op_packages: Optional[List[OpPackageIdentifier]] = None

    """
    Provide a comma-separated list of op packages, interface
    providers, and, optionally, targets to register. Valid values
    for target are CPU and HTP.
    """

    perf_profile: Optional[PerfProfile] = None
    """
    Specifies performance profile to be used. Valid settings are
    low_balanced, balanced, default, high_performance, sustained_high_performance, burst,
    low_power_saver, power_saver, high_power_saver, extreme_power_saver and system_settings.
    """

    synchronous: Optional[bool] = None
    """
    Specifies the way graphs should be executed.
    """

    batch_multiplier: Optional[str] = None
    """
    Specifies the value with which the batch value in input and
    output tensors dimensions will be multiplied. The modified input and output tensors will be
    used only during the execute graphs.
    """

    set_output_tensors: Optional[List[str]] = None
    """
    List of intermediate output tensor names,
    for which the outputs will be written in addition to final graph output tensors.
    Note that options debug and set_output_tensors are mutually exclusive.
    """

    platform_options: Optional[Union[str, Dict[str, str]]] = None
    """
    Specifies values to pass as platform options.
    """

    profiling_level: Optional[ProfilingLevel] = None
    """ Profiling levels: options are "basic", "backend", "detailed" and "client".
        This field should be set within a profiler context. """

    profiling_option: Optional[ProfilingOption] = None
    """ Profiling options: "optrace. This field should be set within a profiler context. """

    use_mmap: Optional[bool] = False
    """
    Specifies whether to use mmap for memory allocation.
    """

    log_level: Optional[str] = None
    """
    Log level for the executor. Standard logging levels are supported.
    """

    duration: Optional[float] = None
    """
    Specifies the duration of the graph execution in seconds.
    Loops over the input_list until this amount of time has transpired.
    """

    num_inferences: Optional[int] = None
    """
    Specifies the number of inferences. Loops over the input_list until
    the specified number of inferences has transpired.
    """

    optional_output_tensor_names: Optional[Union[List[str], Dict[str, Optional[List[str]]]]] = None
    """
    Controls which optional output tensors to return. Mandatory outputs are always included.
    For single-graph models: None returns all optional outputs; an empty list returns no optional outputs,
    and a list of names returns only the specified optional outputs.
    For multi-graph models: Provide a dict mapping graph names to a output tensor (None, empty list,
    or list of names).
    """

    keep_num_outputs: Optional[int] = None
    """
    Specifies the number of outputs to be saved. Once the number of outputs reach the limit,
    subsequent outputs would be discarded.
    """

    def model_post_init(self, __context):
        # Change log level if it is set to None
        if self.log_level is None:
            self._set_log_level()

    @model_validator(mode="after")
    def _validate_num_inference_and_duration_exclusivity(self):
        """Validate that num_inferences and duration are not set simultaneously."""
        if self.num_inferences and self.duration:
            raise ValueError("num_inferences and duration cannot be set simultaneously.")
        return self

    @field_validator("keep_num_outputs", mode="after")
    @classmethod
    def _validate_keep_num_outputs(cls, value):
        """Validate keep_num_outputs is a positive integer."""
        if value and value <= 0:
            raise ValueError("keep_num_outputs must be a positive integer.")
        return value

    @model_validator(mode="after")
    def _validate_input_tensor_arguments(self):
        """Validate that use_native_input_files and native_input_tensor_names are not
        set simultaneously."""
        if self.use_native_input_data and self.native_input_tensor_names:
            raise ValueError(
                "use_native_input_files and native_input_tensor_names cannot be set simultaneously."
            )
        return self

    @model_validator(mode="after")
    def _validate_set_output_tensors_and_debug_arguments(self):
        """Validate that set_output_tensors and debug are not set simultaneously."""
        if self.set_output_tensors and self.debug:
            self.debug = False
            _execute_config_logger.warning(
                "The 'set_output_tensors' and 'debug' option cannot be set simultaneously. "
                "'debug' option has been changed to false."
            )
        return self

    def _set_log_level(self):
        # Import here to avoid global import issues
        from qti.aisw.tools.core.utilities.qairt_logging import QAIRTLogger

        self.log_level = QAIRTLogger.get_default_logging_level("qairt.execute").lower()
