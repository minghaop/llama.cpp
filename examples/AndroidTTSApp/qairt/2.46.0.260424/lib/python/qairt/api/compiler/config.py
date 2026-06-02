# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import enum
from os import PathLike
from typing import List, Literal, Optional, Union, cast

from pydantic import (
    field_validator,
    model_serializer,
    model_validator,
)

from qairt.api.common.backends.aic import AicCompilerConfig
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
from qairt.api.compiler.config_util import get_config_api_options_dict
from qairt.api.configs.common import (
    AISWBaseModel,
    BackendType,
    DspArchitecture,
    OpPackageIdentifier,
    PerfProfile,
)
from qairt.api.configs.device import SocDetails, populate_soc_details_from_factory, soc_details_from_str


class CompileConfig(AISWBaseModel):
    """
    Configuration for AOT compilation.

    Example:

    .. code-block:: python

       config_sm8550 = CompileConfig(backend="HTP", soc_details="chipset:SM8550")

       htp_graph_config = HtpGraphConfig(name="graph_A", optimization_type=3)
       config_O3 = CompileConfig(backend="HTP", graph_custom_configs=[htp_graph_config])

    .. note::

        Custom Configs -  Each <prefix>_custom_configs maps to backend specific settings that can be
        set for QNN API components. Each backend has its own suite of customizable settings.
        Use :func:`qairt.api.common.backends.htp.list_options` to see the list of available options
        for HTP.

    """

    backend: BackendType | str

    context_custom_configs: Optional[List[HtpContextConfig] | List[HtpMcpContextConfig]] = None
    """
    Context configuration options specific to a backend.
    Only HTP and HTP MCP backend options are supported.
    See :class:`qairt.api.common.backends.htp.config.HtpContextConfig` and
    :class:`qairt.api.common.backends.htp_mcp.config.HtpMcpContextConfig` for options
    """

    debug: bool = False
    """
    Enable debug mode.
    """

    device_custom_configs: Optional[List[HtpDeviceConfig] | List[HtpMcpDeviceConfig]] = None
    """
    Device configuration options specific to a backend.
    Only HTP and HTP MCP backend options are supported.
    See :class:`qairt.api.common.backends.htp.config.HtpDeviceConfig`
    and :class:`qairt.api.common.backends.htp_mcp.config.HtpMcpDeviceConfig` for options.
    """

    graph_custom_configs: Optional[List[HtpGraphConfig] | List[HtpMcpGraphConfig]] = None
    """
    Graph configuration options specific to a backend.
    Currently only used for HTP backend. See :class:`qairt.api.common.backends.htp.config.HtpGraphConfig`
    and :class:`qairt.api.common.backends.htp_mcp.config.HtpMcpGraphConfig` for options.
    """

    # TODO: Set default to None after AISW-139520
    log_level: Optional[str] = "error"
    """
    Log level for the compiler. Standard logging levels are supported.
    """

    op_packages: Optional[List[OpPackageIdentifier]] = None
    """
    List of custom op package identifiers to be used for compilation.
    """

    compiler_custom_configs: Optional[AicCompilerConfig] = None
    """
    Set this field to enable configurations that are passed by the backend to the compiler.
    Note this option is currently only applicable to the AIC Backend.
    Use :class:`qairt.api.common.backends.aic.config.AicCompilerConfig.list_config_options` to see valid fields.
    """

    # Output Options
    enable_intermediate_outputs: Optional[bool] = None
    """
    Enable all intermediate nodes to be produced along with default outputs in the saved context.
    Note that options enable_intermediate_outputs and set_output_tensors are mutually exclusive.
    Only one of the options can be specified at a time.
    """

    set_output_tensors: Optional[List[str]] = None
    """
    A comma-separated list of intermediate output tensor names, for which the outputs
    will be written in addition to final graph output tensors.  The syntax is: graphName0:tensorName0,
    tensorName1;graphName1:tensorName0,tensorName1. In case of a single graph, its name is not necessary and
    a list of comma separated tensor names can be provided, e.g.: tensorName0,tensorName1
    """

    soc_details: Optional[Union[SocDetails, str]] = None
    """
    Device specification to use for compilation. Can be specified as a spec string
    in the form "chipset:value;dsp_arch:value;soc_model:value|...". This option
    will be ignored if any device custom configurations are set.
    """

    io_tensor_mem_type: Optional[Literal["raw", "memhandle"]] = "raw"
    """
    Select memory type for input or output tensors. Possible options are: "raw" and "memhandle".
    """

    memory_custom_config: Optional[HtpMemoryConfig] = None
    """
    Memory backend configuration for the compiler. Only HTP backend configurations are supported.
    """

    data_format_config: Optional[Union[str, PathLike]] = None
    """
    Path to a JSON config file, specifying the data formats of certain tensors.
    """

    group_context_config: Optional[HtpGroupContextConfig] = None
    """
    Group context configuration for the compiler. Only HTP backend configurations are supported.
    """

    backend_binary_path: Optional[str] = None
    """
    Name or path of the binary file to save a backend-specific context binary to with .bin file extension.
    The file name is expected to have the suffix _backend. If not provided, the suffix would be added.
    File name can be, for example, myqpc.bin or myqpc (without the .bin extension).
    If not provided, no backend binary is created. If absolute path is provided, binary is saved in this path.
    """

    _accepts_profiling_args: bool = True

    _soc_details_resolved: Optional[SocDetails] = None

    def model_post_init(self, __context):
        # populate attributes related to soc details
        if self.soc_details and not self.device_custom_configs:
            self._set_soc_details()

        # Change log level if it is set to None
        if self.log_level is None:
            self._set_log_level()

    @field_validator("backend", mode="after")
    @classmethod
    def _validate_backend_type(cls, value):
        """Validate that backend type is offline preparable"""
        if not BackendType.is_valid_backend(value):
            raise ValueError(f"Invalid backend type: {value}")
        if value not in BackendType.offline_preparable_backends():
            raise ValueError(f"Backend type {value} is not supported for offline compilation.")
        return BackendType(value)

    @field_validator("compiler_custom_configs", mode="after")
    @classmethod
    def _validate_compiler_custom_configs(cls, value, values):
        """Validates that compiler_custom_configs is used only with AIC backend"""
        if value and values.data["backend"] != BackendType.AIC:
            raise ValueError("compiler_custom_configs is only supported with AIC backend.")
        return value

    @model_validator(mode="after")
    def _validate_output_tensor_arguments(self):
        """Validate that enable_intermediate_outputs and set_output_tensors are not
        set simultaneously."""
        if self.enable_intermediate_outputs and self.set_output_tensors:
            raise ValueError(
                "enable_intermediate_outputs and set_output_tensors cannot be set simultaneously."
            )
        return self

    @model_validator(mode="after")
    def _validate_extended_udma(self):
        """Validate that extended_udma is only enabled when the HTP device config has dsp_arch >= v81."""
        ctx_cfg = (
            self.context_custom_configs[0]
            if self.context_custom_configs and isinstance(self.context_custom_configs[0], HtpContextConfig)
            else None
        )
        dev_cfg = (
            self.device_custom_configs[0]
            if self.device_custom_configs and isinstance(self.device_custom_configs[0], HtpDeviceConfig)
            else None
        )
        if ctx_cfg and ctx_cfg.extended_udma and dev_cfg and dev_cfg.dsp_arch is not None:
            arch_version = int(str(dev_cfg.dsp_arch).lstrip("v"))
            if arch_version < 81:
                raise ValueError(
                    f"extended_udma is only supported on HTP v81 and above, "
                    f"but dsp_arch '{dev_cfg.dsp_arch}' was found."
                )
        return self

    def _set_soc_details(self):
        """Sets soc details and uses it to derive device custom configs"""
        if not self.backend == BackendType.HTP:
            print(f"Compile time device specifications are only valid for HTP. Ignoring device specification")
            return self

        if isinstance(self.soc_details, str):
            soc_details = soc_details_from_str(self.soc_details)
        else:
            soc_details = self.soc_details

        # This is only used when device_custom_configs is not set.
        device_custom_configs: List[HtpDeviceConfig] = []
        device_id = 0

        # Try to populate the soc model field if unset
        if not soc_details.model or not soc_details.dsp_arch:
            if not populate_soc_details_from_factory(soc_details):
                print(
                    f" Could not set soc model for chipset: {soc_details.chipset}. "
                    f" Skipping device config creation. Please set soc model"
                    f" and dsp arch manually."
                )
                return self

        # set soc details that are mapped
        self._soc_details_resolved = soc_details

        if self.device_custom_configs:
            # Device custom configs are already set
            print(f" Device custom configs are already set. Skipping device config creation from soc details")
            return

        if "|" in soc_details.model:
            soc_models = soc_details.model.split("|")  # multiple socs are requested
        else:
            soc_models = [soc_details.model]

        for soc_model in soc_models:
            # dsp arch is retrieved as an int from the device factory
            dsp_arch = str(soc_details.dsp_arch)
            if "v" not in dsp_arch:
                dsp_arch = "v" + dsp_arch

            n = soc_details.num_cores or 1
            cores = [HtpDeviceCoreConfig(id=i) for i in range(n)]

            # prepopulate device custom configs
            device_custom_configs.append(
                HtpDeviceConfig(
                    id=device_id,
                    dsp_arch=dsp_arch,
                    soc_model=int(soc_model),
                    cores=cores,
                )
            )
            device_id += 1

        self.device_custom_configs = device_custom_configs

    def _set_log_level(self):
        # Import here to avoid global import issues
        from qti.aisw.tools.core.utilities.qairt_logging import QAIRTLogger

        self.log_level = QAIRTLogger.get_default_logging_level("qairt.compile").lower()

    def set_mode(self, mode: str, **kwargs):
        """
        Define a mode for compilation. Only weight sharing is supported.

        This option is most effective when no custom configs have been set and soc details are set with the
        chipset.

        Args:
            mode: Mode to be set. Currently only weight_sharing is supported.
            kwargs: Keyword arguments for the mode. For mode == weight_sharing, see
                    :func:`qairt.api.compiler.config.CompilerModeSetters.set_weight_sharing` for
                    the arguments that can be provided.

        """

        # Only weight sharing is supported. More configuration modes will be added in the future.
        if mode == CompilerModes.WEIGHT_SHARING.value:
            CompilerModeSetters.set_weight_sharing(self, **kwargs)
        else:
            # TODO: add GPU and other backend tuning modes
            raise ValueError(f"Invalid mode: {mode}. Only weight_sharing is currently supported.")

        return self

    @model_serializer
    def serialize(self, info):
        """Serialize the model to a dictionary. Set backend_extensions to True to serialize
        only backend extensions. Use config.model_dump() to call this function indirectly.

        Examples:

            .. code-block:: python

                from qairt.api.compiler.config import CompilerConfig

                # to dump the entire config
                config_dict = config.model_dump()

                # to dump just the backend extensions
                config_dict = config.model_dump(context={"backend_extensions": True})

        """

        context = info.context or {}

        if context.get("backend_extensions", False):
            # dump only backend extensions
            qnn_api_config_options = get_config_api_options_dict(self)
            backend_custom_config_dict = qnn_api_config_options["backend_extensions"]["config_dict"]

            return backend_custom_config_dict

        return self.__dict__


class CompilerModes(enum.Enum):
    WEIGHT_SHARING = "weight_sharing"

    @staticmethod
    def list_options():
        return [mode.value for mode in CompilerModes]


class CompilerModeSetters:
    """Class to set different compilation modes."""

    @staticmethod
    def set_weight_sharing(config: CompileConfig, **kwargs):
        """
        Sets graph, device and core config options to enable weight sharing on the HTP backend.

        Expects the config to contain soc details which have been resolved.

        Arguments:

           config: The Compile config instance

           graph_names: A list of graph names to be used for weight sharing. If this is not provided
                then "ensure_graphs" will be set in the config. This property will signal to other
                APIs that the graph names should be inferred from the model.
           soc_model: The soc model to use for weight sharing. If this is not provided then the soc model
              will be inferred from the soc details.
           dsp_arch: The dsp arch to use for weight sharing. If this is not provided then the dsp arch
                 will be inferred from the soc details.
           hvx_threads: The number of hvx threads to associate with the dsp arch
           vtcm_size_in_mb: The vtcm size in megabytes
           fp16: A boolean indicate if fp16 should be enabled

        """
        if config.backend != BackendType.HTP:
            raise ValueError(
                f" Weight sharing invalid for backend: {config.backend}. Only HTP backend is supported."
            )

        # set io_tensor_mem_type to memhandle by default
        if not config.io_tensor_mem_type or config.io_tensor_mem_type == "raw":
            config.io_tensor_mem_type = "memhandle"

        if not config.graph_custom_configs:
            graph_names = kwargs.get("graph_names", None)
            if not graph_names:
                graph_names = ["", ""]  # set empty graph names to be filled in later
                config.__dict__["ensure_graphs"] = True

            # retrieve soc details if available
            soc_details: SocDetails | None = config._soc_details_resolved

            hvx_threads = 0
            vtcm_size_in_mb = 0
            fp16 = False

            if soc_details is not None:
                hvx_threads = soc_details.num_of_hvx_threads
                vtcm_size_in_mb = soc_details.vtcm_size_in_mb
                fp16 = soc_details.supports_fp16

            # if values were not set in kwargs, then soc details will be used if soc
            # details are not None, otherwise defaults will be used.
            hvx_threads = kwargs.get("hvx_threads", hvx_threads)
            vtcm_size_in_mb = kwargs.get("vtcm_size_in_mb", vtcm_size_in_mb)
            fp16 = kwargs.get("fp16", fp16)

            config.graph_custom_configs = [
                HtpGraphConfig(
                    name=name,
                    optimization_type=3,
                    fp16_relaxed_precision=fp16,
                    hvx_threads=hvx_threads,
                    vtcm_size_in_mb=vtcm_size_in_mb,
                )
                for name in graph_names
            ]

        if not config.context_custom_configs:
            config.context_custom_configs = [HtpContextConfig()]

        for ctx_cfg in config.context_custom_configs:
            assert isinstance(ctx_cfg, HtpContextConfig)
            ctx_cfg.weight_sharing_enabled = True

        if config.device_custom_configs is None:
            # Ensure properties needed for weight sharing are set
            if not hasattr(kwargs, "dsp_arch") or not hasattr(kwargs, "soc_model"):
                print("No dsp arch provided. Defaulting to dsp_arch: v79 and soc_model: 69")

            core_device_config = HtpDeviceCoreConfig(perf_profile=PerfProfile.BURST)
            # id = 0, rpc_control_latency = 100
            config.device_custom_configs = [
                HtpDeviceConfig(
                    soc_model=kwargs.get("soc_model", 69),
                    dsp_arch=kwargs.get("dsp_arch", "v79"),
                    cores=[core_device_config],
                )
            ]
        else:
            for idx in range(len(config.device_custom_configs)):
                cfg = config.device_custom_configs[idx]
                if isinstance(cfg, HtpDeviceConfig) and hasattr(cfg, "cores"):
                    for core_idx, core in enumerate(cfg.cores):
                        if core.perf_profile != PerfProfile.BURST:
                            cfg.cores[core_idx].perf_profile = PerfProfile.BURST
                else:
                    # Only HTP has a core performance setting
                    continue

        # handle memory config, set to shared buffer by default
        if not config.memory_custom_config:
            config.memory_custom_config = HtpMemoryConfig()
