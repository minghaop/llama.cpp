# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from uuid import uuid4

from qairt.api.configs.common import BackendType
from qairt.modules.genie_execution.genie_config import (
    DialogType,
    EagletDialog,
    EagletDraftDialogEngine,
    EngineModelType,
    GenieConfig,
    SSDDialog,
)
from qairt.utils import loggers
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    ConnectionType,
    DevicePlatformType,
    RemoteDeviceInfo,
)
from qti.aisw.tools.core.utilities.devices.api.device_factory import DeviceFactory
from qti.aisw.tools.core.utilities.devices.api.device_interface import DeviceInterface

logger = loggers.get_logger(name=__name__)


class GenieT2TRunnerHelper:
    @staticmethod
    def resolve_artifact_path(
        artifact_path: str | os.PathLike,
        local_config_artifacts: Dict[str | os.PathLike, str | os.PathLike],
        target_name: str | None = None,
    ) -> str | os.PathLike:
        """
        Resolve the target path for a configuration artifact with a unique name to avoid collisions.

        Args:
            artifact_path: Source path of the artifact.
            local_config_artifacts: Mapping of target paths to source paths (updated in-place).
            target_name: Optional explicit filename to use in the artifacts directory.
                If omitted, the basename of ``artifact_path`` is used.
        """
        target_sep = "/"  # Genie can handle Unix-style path separators on all targets
        name = target_name if target_name is not None else os.path.basename(artifact_path)
        target_path = target_sep.join(["artifacts", name])

        # To avoid naming conflicts, prepend a random string if a conflict occurs
        while target_path in local_config_artifacts:
            target_path = target_sep.join(["artifacts", str(uuid4())[:6] + name])
        local_config_artifacts[target_path] = artifact_path
        return target_path

    @staticmethod
    def prepare_config(config: GenieConfig) -> Tuple[GenieConfig, Dict[str | os.PathLike, str | os.PathLike]]:
        """
        Prepare a Genie configuration for deployment on a device.

        This method validates the configuration, resolves all required artifact paths.
        It returns the possibly‑modified configuration and a mapping of artifact destinations
        to their original source paths.
        """
        local_config_artifacts: Dict[str | os.PathLike, str | os.PathLike] = {}
        local_config = config
        target_sep = "/"  # Genie can handle Unix-style path separators on all targets

        if not os.path.exists(config.dialog.tokenizer.path):
            raise ValueError(
                f"Invalid tokenizer path in provided Genie config: {config.dialog.tokenizer.path}"
            )

        local_config.dialog.tokenizer.path = GenieT2TRunnerHelper.resolve_artifact_path(
            config.dialog.tokenizer.path, local_config_artifacts
        )

        self_engines = sorted(
            [local_config.dialog.engine]
            if not isinstance(local_config.dialog.engine, list)
            else local_config.dialog.engine,
            key=lambda e: type(e).__name__,
        )
        config_engines = sorted(
            [config.dialog.engine] if not isinstance(config.dialog.engine, list) else config.dialog.engine,
            key=lambda e: type(e).__name__,
        )

        for self_engine, config_engine in zip(self_engines, config_engines):
            if self_engine.model.type == EngineModelType.LIBRARY.value:
                assert self_engine.model.library is not None
                assert config_engine.model.library is not None
                self_engine.model.library.model_bin = GenieT2TRunnerHelper.resolve_artifact_path(
                    config_engine.model.library.model_bin, local_config_artifacts
                )

            if (
                self_engine.model.type == EngineModelType.BINARY.value
                and config_engine.model.binary
                and self_engine.model.binary
            ):
                for i, ctx_bin in enumerate(config_engine.model.binary.ctx_bins):
                    if not os.path.exists(ctx_bin):
                        raise ValueError(
                            f"Context binary path provided in genie config does not exist: {ctx_bin}"
                        )
                    self_engine.model.binary.ctx_bins[i] = GenieT2TRunnerHelper.resolve_artifact_path(
                        ctx_bin, local_config_artifacts, target_name=f"split_model_{i + 1}.bin"
                    )

                if config_engine.backend.extensions:
                    if not os.path.exists(config_engine.backend.extensions):
                        raise ValueError(
                            "Backend Extensions config path provided in genie config does not exist: "
                            f"{config_engine.backend.extensions}"
                        )
                    self_engine.backend.extensions = GenieT2TRunnerHelper.resolve_artifact_path(
                        config_engine.backend.extensions, local_config_artifacts
                    )
                if config_engine.model.positional_encoding:
                    self_engine.model.positional_encoding = config_engine.model.positional_encoding
            if self_engine.model.binary and self_engine.model.binary.lora:
                if self_engine.model.binary.lora.adapters:
                    for i, adapter in enumerate(self_engine.model.binary.lora.adapters):
                        bin_sections = adapter.bin_sections if adapter.bin_sections else []
                        for j, binary in enumerate(bin_sections):
                            if binary == "":
                                pass
                            else:
                                if not os.path.exists(binary):
                                    raise ValueError(f"Provided adapter binary path does not exist: {binary}")

                                self_engine.model.binary.lora.adapters[i].bin_sections[j] = (
                                    GenieT2TRunnerHelper.resolve_artifact_path(binary, local_config_artifacts)
                                )

        if (
            isinstance(config.dialog, SSDDialog)
            and isinstance(local_config.dialog, SSDDialog)
            and config.dialog.ssd_q1
            and local_config.dialog.ssd_q1
        ):
            _SSD_KVCACHE_PREFIX_FILENAME = "kv-cache.primary.qnn-htp"
            if not (
                os.path.isdir(config.dialog.ssd_q1.forecast_prefix_name)
                and os.path.exists(
                    os.path.join(config.dialog.ssd_q1.forecast_prefix_name, _SSD_KVCACHE_PREFIX_FILENAME)
                )
            ):
                provided_path = (
                    local_config.dialog.ssd_q1.forecast_prefix_name if local_config.dialog.ssd_q1 else None
                )
                raise ValueError(
                    "forecast-prefix-name must point to a directory containing the forecast prefix file"
                    f"named: {_SSD_KVCACHE_PREFIX_FILENAME}. Invalid path provided: "
                    f"{provided_path}"
                )
            src_path = os.path.join(config.dialog.ssd_q1.forecast_prefix_name, _SSD_KVCACHE_PREFIX_FILENAME)
            target_path = target_sep.join(["ssd", _SSD_KVCACHE_PREFIX_FILENAME])
            local_config_artifacts[target_path] = src_path
            local_config.dialog.ssd_q1.forecast_prefix_name = "ssd"
        if isinstance(config.dialog, EagletDialog):
            dest_draft_engine = next(
                engine for engine in local_config.dialog.engine if isinstance(engine, EagletDraftDialogEngine)
            )
            source_draft_engine = next(
                engine for engine in config.dialog.engine if isinstance(engine, EagletDraftDialogEngine)
            )
            if source_draft_engine.model.draft_token_map:
                dest_draft_engine.model.draft_token_map = GenieT2TRunnerHelper.resolve_artifact_path(
                    source_draft_engine.model.draft_token_map, local_config_artifacts
                )

        if config.dialog.embedding:
            assert local_config.dialog.embedding is not None
            local_config.dialog.embedding.lut_path = GenieT2TRunnerHelper.resolve_artifact_path(
                config.dialog.embedding.lut_path, local_config_artifacts
            )

        return local_config, local_config_artifacts

    @staticmethod
    def get_sdk_artifacts(
        target_device: DeviceInterface, qairt_sdk_root, backend_type: BackendType
    ) -> Tuple[list, list]:
        """
        Determine the SDK binaries and libraries that need to be pushed to the device.

        For remote devices (non‑local and non‑Android) this method constructs the list
        of required binaries and libraries based on the target platform and backend
        type. It returns two lists: one for binary artifacts and one for library
        artifacts. Local devices do not require any SDK artifacts to be pushed.
        """
        libs_to_push = []
        bins_to_push = []
        assert target_device.device_info is not None
        if (
            target_device.device_info.connection_type == ConnectionType.LOCAL
            and target_device.device_info.platform_type != DevicePlatformType.ANDROID
        ):
            return [], []  # Local device does not need to push SDK artifacts

        if target_device.device_info.platform_type == DevicePlatformType.ANDROID:
            target_name = "aarch64-android"
        elif target_device.device_info.platform_type == DevicePlatformType.X86_64_LINUX:
            target_name = "x86_64-linux-clang"
        elif target_device.device_info.platform_type == DevicePlatformType.LINUX_EMBEDDED:
            target_name = "aarch64-oe-linux-gcc11.2"
        else:
            raise ValueError(f"Unsupported platform: {target_device.device_info.platform_type}")

        bins_to_push.append(os.path.join(qairt_sdk_root, "bin", target_name, "genie-t2t-run"))
        libs_to_push.append(os.path.join(qairt_sdk_root, "lib", target_name, "libGenie.so"))
        libs_to_push.append(os.path.join(qairt_sdk_root, "lib", target_name, "libQnnSystem.so"))

        from qti.aisw.tools.core.modules.api.utils.configure_backend import Target, create_backend

        identifier = None
        credentials = None
        if isinstance(target_device.device_info, RemoteDeviceInfo):
            identifier = target_device.device_info.identifier
            credentials = target_device.device_info.credentials

        backend = create_backend(
            backend_type.value,
            Target(
                type=target_device.device_info.platform_type,
                identifier=identifier,
                credentials=credentials,
            ),
        )

        if backend.backend_library:
            libs_to_push.append(os.path.join(qairt_sdk_root, "lib", target_name, backend.backend_library))

        libs_to_push.extend(backend.get_required_device_artifacts(str(qairt_sdk_root)))

        if backend.backend_extensions_library:
            libs_to_push.append(
                os.path.join(qairt_sdk_root, "lib", target_name, backend.backend_extensions_library)
            )
        if backend_type == "CPU":
            libs_to_push.append(os.path.join(qairt_sdk_root, "lib", target_name, "libQnnGenAiTransformer.so"))
            libs_to_push.append(
                os.path.join(qairt_sdk_root, "lib", target_name, "libQnnGenAiTransformerCpuOpPkg.so")
            )
            libs_to_push.append(
                os.path.join(qairt_sdk_root, "lib", target_name, "libQnnGenAiTransformerModel.so")
            )
        return bins_to_push, libs_to_push
