# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import pathlib
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Union, overload

import yaml  # type: ignore

from qairt.api.compiled_model import CompiledModel
from qairt.api.compiler.config import CompileConfig
from qairt.api.compiler.config_util import get_config_api_options_dict
from qairt.api.configs.common import BackendType, ProfilingData
from qairt.api.model import Model
from qairt.api.profiler.profiler import profile
from qairt.modules.cache_module import CacheModule
from qairt.modules.lora.lora_config import serialize_lora_adapter_weight_config
from qairt.modules.qti_import import qti_module_api
from qairt.utils.asset_utils import Asset, AssetType, check_asset_type
from qairt.utils.exceptions import CompilationError
from qairt.utils.loggers import get_logger, maybe_suppress_stdout
from qti.aisw.tools.core.modules.context_bin_gen import ContextBinGen, ContextBinGenArgConfig, GenerateConfig

_compile_logger = get_logger("qairt.compile")


@overload
def compile(model: Union[Model, List[Model]], backend: str | BackendType) -> CompiledModel: ...


@overload
def compile(model: Union[Model, List[Model]], config: CompileConfig) -> CompiledModel: ...


@profile("compile")
def compile(
    model: Union[Model, List[Model]],
    *,
    backend: Optional[str | BackendType] = None,
    config: Optional[CompileConfig] = None,
) -> CompiledModel:
    """
    Compile the given model for a specified backend using optional configuration options.

    Args:
        model(Union[Model, List[Model]]): The model(s) object to compile. If a list of models is provided, the
            models will be compiled into a single compiled model.
        backend(Optional[str | BackendType]): The QAIRT backend for which to compile the model e.x. "HTP", "AIC".
            Use :func:`qairt.api.configs.common.BackendType.offline_preparable_backends()` for the full list of supported backends.
        config(Optional[CompileConfig]): The compile configuration to use. See :class:`qairt.api.compiler.config.CompileConfig`
            for the full list of options.

    Examples:
        .. code-block:: python

            # Compile a model to HTP backend
            compiled_model = qairt.compile(my_model, backend="HTP")


            # Compile the model for a particular SOC
            compile_config =  CompileConfig(backend="HTP", soc_details="chipset:SM8550")
            compiled_model_8550 = qairt.compile(my_model, config=compile_config)


    Returns:
        CompiledModel: A new CompiledModel instance containing a reference to a generated context binary.

    Raises:
        CompilationError: If a context binary could not be generated from the Model.

    .. note:: One of backend and config should be specified. if both backend and config are specified,
              the backend argument will be ignored.

    """

    if config is not None:
        backend = config.backend
    elif backend is None:
        raise ValueError("Either backend or config must be specified")

    def validate_and_save_model(_model):
        if isinstance(_model, CompiledModel):
            raise TypeError(
                f" Compile is not a valid operation on a compiled model: {_model.name}. "
                f" Expected instance of type: {Model.__name__}, got {CompiledModel.__name__}"
            )

        # Models need to be saved before binary generation can be called
        if not _model.module.path:
            _model.save()

    shared_context_models: list[Model] = []
    if isinstance(model, list):
        validate_and_save_model(model[0])

        if len(model) > 1:
            for sm in model[1:]:
                validate_and_save_model(sm)
                shared_context_models.append(sm)
        model = model[0]
    else:
        validate_and_save_model(model)

    if config is None:
        config = CompileConfig(backend=backend)

    # Set Generate Config
    adapter_weight_config_file = None

    def _generate_weight_config(model: Model) -> pathlib.Path:
        if isinstance(model.lora_use_cases, (str, os.PathLike)):
            return pathlib.Path(model.lora_use_cases)

        elif isinstance(model.lora_use_cases, list):
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_yaml_file:
                yaml_path = temp_yaml_file.name

            # Ensure base_dir is correctly derived
            base_dir = str(getattr(model.module, "path", ""))

            # Serialize the config
            serialize_lora_adapter_weight_config(model.lora_use_cases, yaml_path, base_dir)
            return pathlib.Path(yaml_path)

        else:
            raise TypeError(f"Unexpected type for lora_use_cases: {type(model.lora_use_cases)}")

    if model.lora_use_cases:
        adapter_weight_config_file = _generate_weight_config(model)
        if shared_context_models:
            # TODO: Shouldn't need to serialize all these just to read them in again and then serialize once again
            merged_use_cases = []
            adapter_config_paths = [adapter_weight_config_file] + [
                _generate_weight_config(_model) for _model in shared_context_models
            ]
            for adapter_config_path in adapter_config_paths:
                with adapter_config_path.open("r") as f:
                    adapter_config_data = yaml.safe_load(f)
                merged_use_cases.extend(adapter_config_data["use_case"])

            merged_adapter_config_data = {
                "use_case": merged_use_cases,
                "share_adapters_between_graphs": "Yes",
            }
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_yaml_file:
                merged_adapter_config_path = pathlib.Path(temp_yaml_file.name)

            with merged_adapter_config_path.open("w") as f:
                yaml.dump(merged_adapter_config_data, f, default_flow_style=False, sort_keys=False)
            adapter_weight_config_file = merged_adapter_config_path

    generate_config = GenerateConfig(
        input_output_tensor_mem_type=config.io_tensor_mem_type,
        set_output_tensors=config.set_output_tensors,
        enable_intermediate_outputs=config.enable_intermediate_outputs,
        profiling_level=getattr(config, "profiling_level", None),
        profiling_option=getattr(config, "profiling_option", None),
        log_level=config.log_level,
        op_packages=config.op_packages,
        adapter_weight_config_file=adapter_weight_config_file,
        data_format_config=config.data_format_config,
    )

    # attach any shared context models
    ctx_bin_arg_models = [qti_module_api.ModelConfig(path=Path(model.module.path))]

    if shared_context_models:
        # Ensure graphs names are set for shared context models
        # This can only be enabled via config.set_mode
        all_models = [model, *shared_context_models]

        _compile_logger.debug(f"Detected multiple models: {(model.name for model in all_models)}")

        # This code broadcasts the same graph configuration across multiple models
        # if ensure graphs is set
        if getattr(config, "ensure_graphs", False) and config.graph_custom_configs is not None:
            base_graph_config = config.graph_custom_configs[0]
            config.graph_custom_configs = []
            for md in all_models:
                graphs_info = md.module.graphs_info
                for gi in graphs_info:
                    shared_config: Any = base_graph_config.model_copy()
                    shared_config.name = gi
                    config.graph_custom_configs.append(shared_config)

        # attach any shared context models
        ctx_bin_arg_models += [
            qti_module_api.ModelConfig(path=Path(scm.module.path)) for scm in shared_context_models
        ]

    # retrieve API config options
    qnn_api_config_options = get_config_api_options_dict(config)
    backend_custom_config_dict = qnn_api_config_options["backend_extensions"]["config_dict"]
    _compile_logger.debug(f"Backend custom options set to: {backend_custom_config_dict}")

    # Create ContextBinGenArgConfig
    root_dir = os.getenv("QAIRT_TMP_DIR", default=tempfile.gettempdir())
    tmp_dir = Path(tempfile.mkdtemp(prefix="temp_working_dir_", dir=root_dir))

    backend_binary_path = None
    if config.backend_binary_path:
        backend_binary_path = (
            os.path.splitext(config.backend_binary_path)[0]
            if str(config.backend_binary_path).endswith(".bin")
            else config.backend_binary_path
        )
        # Adding '_backend' suffix if not provided by user
        if not str(backend_binary_path).endswith("_backend"):
            backend_binary_path += "_backend"
        backend_binary_path = os.path.join(tmp_dir, os.path.basename(backend_binary_path))
    ctx_arg_config = ContextBinGenArgConfig(
        backend=backend,
        model=ctx_bin_arg_models,
        backend_config_dict=backend_custom_config_dict,
        generate_config=generate_config,
        output_filename=model.name,
        output_dir=tmp_dir,
        backend_specific_filename=backend_binary_path,
    )

    # Init context binary generator
    context_bin_gen = ContextBinGen(logger=_compile_logger)

    # We need this context manager because HTP prepare does not honor logging levels cleanly
    # and we get a ton of warnings even if we set the level to critical.
    # So we depend on the compile logger. If it is not set to debug, then we just suppress
    # the output entirely expecting an error to write to stderr instead
    with maybe_suppress_stdout(logger=_compile_logger):
        out_config = context_bin_gen.generate(ctx_arg_config)
    try:
        # load cache module from generated path
        cache_module = CacheModule.load(path=out_config.context_binary.path)
        cache_module.working_directory = tmp_dir  # delete this cache if it is never saved
    except Exception as e:
        _compile_logger.debug(f"Could not load context binary for {model.name}: {e}")
        raise CompilationError(f"Could not create compiled binary for {model.name}")

    # create compiled model
    # TODO: This should be creating a dlc module and populating it with cache info
    # Doing this temporarily until DLC Interactions API is fully developed
    # JIRA: https://jira-dc.qualcomm.com/jira/browse/AISW-121951
    compiled_model = CompiledModel(cache_module, backend, config=config)

    _compile_logger.info(f"Compiled model: {model.name} for backend: {backend}")

    if out_config.profiling_data:
        _handle_compile_profile_data(compiled_model, out_config)

    if out_config.backend_binary_path:
        backend_binary = Path(out_config.backend_binary_path.path)
        check_asset_type(AssetType.BACKEND_BIN, backend_binary)
        compiled_model.assets[backend_binary.name] = Asset(path=backend_binary, delete=True)

    if isinstance(model, list):
        if hasattr(model[0], "lora_use_cases"):
            compiled_model.lora_use_cases = model[0].lora_use_cases
    else:
        if hasattr(model, "lora_use_cases"):
            compiled_model.lora_use_cases = model.lora_use_cases

    # Clean up
    if isinstance(model.lora_use_cases, list) and adapter_weight_config_file:
        try:
            os.remove(adapter_weight_config_file)
            _compile_logger.debug(
                f"Deleted temporary adapter weight config file: {adapter_weight_config_file}"
            )
        except Exception as e:
            _compile_logger.warning(f"Failed to delete temporary adapter weight config file: {e}")

    return compiled_model


def _handle_compile_profile_data(compiled_model, out_config):
    profiling_log = Path(out_config.profiling_data.profiling_log)
    # Check and set profiling asset
    check_asset_type(AssetType.PROFILING_LOG, profiling_log)
    setattr(compiled_model, "profiling_data", ProfilingData(profiling_log=profiling_log))

    compiled_model.assets[profiling_log.name] = Asset(path=profiling_log, delete=True)
    _compile_logger.debug(f"Retrieved profiling log: {profiling_log.name}")

    if out_config.profiling_data.backend_profiling_artifacts:
        for artifact in out_config.profiling_data.backend_profiling_artifacts:
            artifact = Path(artifact)
            check_asset_type(AssetType.SCHEMATIC_BIN, artifact)
            compiled_model.assets[artifact.name] = Asset(path=artifact, delete=True)
            _compile_logger.debug(f"Retrieved profiling artifact: {artifact.name}")
