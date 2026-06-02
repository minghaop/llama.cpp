# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
from typing import TYPE_CHECKING, Dict

from pydantic import DirectoryPath

from qairt.api.common.backends.aic.config import AicConfigHelper
from qairt.api.common.backends.htp import HtpContextConfig, HtpDeviceConfig, HtpGraphConfig
from qairt.api.common.backends.htp.config import HtpConfigHelper
from qairt.api.configs.common import BackendType

if TYPE_CHECKING:
    from qairt.api.executor.execution_config import ExecutionConfig


def get_config_api_options_dict(config: "ExecutionConfig") -> Dict:
    """
    Converts the execution config to a dictionary. The dictionary is structured such that it can be
    encoded as a json string for serialization purposes. The dictionary structure is as follows:

    execute_config_dict = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": "",
            "config_dict": {}
        }
        "context_configs": {}
    }

    Args:
        config (ExecutionConfig): Execution config to be converted to a dict.

    Returns:
        Dictionary representation of the execution config.
    """

    backend_extensions_dict = {}
    shared_library_path = ""
    if config.backend and config.backend == BackendType.HTP:
        ctx_custom_configs = config.context_custom_configs or []
        graph_custom_configs = config.graph_custom_configs or []
        device_custom_configs = config.device_custom_configs or []
        backend_extensions_dict = HtpConfigHelper.to_backend_extension_dict(
            context_configs=[
                HtpContextConfig(**cfg.model_dump(exclude_unset=True, exclude_none=True))
                for cfg in ctx_custom_configs
            ],
            graph_configs=[
                HtpGraphConfig(**cfg.model_dump(exclude_unset=True, exclude_none=True))
                for cfg in graph_custom_configs
            ],
            device_configs=[
                HtpDeviceConfig(**cfg.model_dump(exclude_unset=True, exclude_none=True))
                for cfg in device_custom_configs
            ],
            memory_config=config.memory_custom_config,
        )
        shared_library_path = HtpConfigHelper.shared_library_path()
    elif config.backend and config.backend == BackendType.AIC:
        backend_extensions_dict = AicConfigHelper.to_backend_extension_dict(
            runtime_config=config.runtime_custom_config
        )
        shared_library_path = AicConfigHelper.shared_library_path()

    context_config_dict = (
        config.context_execute_custom_config.model_dump() if config.context_execute_custom_config else {}
    )

    execute_config_dict = {
        "backend_extensions": {"shared_library_path": shared_library_path, "config_file_path": ""},
        "context_configs": context_config_dict,
    }

    # add non-schema compliant value of config as a dictionary
    execute_config_dict["backend_extensions"]["config_dict"] = backend_extensions_dict

    return execute_config_dict


# TODO: Files could be avoided on local host
def _get_config_api_options_json_file(
    config: "ExecutionConfig", output_dir: DirectoryPath, prefix: str = ""
) -> str:
    """
    Saves the dictionary output from get_config_api_options_dict into a json file.
    If the backend extensions is a dictionary, that is also dumped to a json file.

    Args:
        config (ExecutionConfig): Execution config to be converted to a dict.
        output_dir (DirectoryPath): Path to directory where the json needs to be dumped.
        prefix (str): String to be added as a prefix to the json file name.

    Returns:
        Path to the output json file as a string.
    """
    if prefix:
        prefix = f"{prefix}_"

    backend = config.backend.lower() if config.backend else ""
    execute_cfg_dict = get_config_api_options_dict(config)
    if backend_cfg := execute_cfg_dict["backend_extensions"]["config_dict"]:
        backend_ext_file_path = output_dir / f"{prefix}{backend}_extensions.json"
        with open(str(backend_ext_file_path), "w+") as fp:
            json.dump(backend_cfg, fp, indent=4)
        execute_cfg_dict["backend_extensions"]["config_file_path"] = str(backend_ext_file_path)

    # remove the config dict from the execute config dict before saving to file
    del execute_cfg_dict["backend_extensions"]["config_dict"]

    execute_config_file_path = output_dir / f"{prefix}execute_config.json"
    with open(str(execute_config_file_path), "w+") as fp:
        json.dump(execute_cfg_dict, fp, indent=4)

    return str(execute_config_file_path)
