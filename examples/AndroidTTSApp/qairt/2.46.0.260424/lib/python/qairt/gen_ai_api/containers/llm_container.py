# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import json
import os
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Union

from qairt import __sdk_version__ as sdk_version
from qairt.api.compiled_model import CompileConfig, CompiledModel
from qairt.api.configs.common import AISWBaseModel, BackendType
from qairt.api.configs.device import Device
from qairt.gen_ai_api.configs.eaglet_config import EagletRunConfig
from qairt.gen_ai_api.configs.gen_ai_config import GenAIConfig
from qairt.gen_ai_api.containers.gen_ai_container import GenAIContainer
from qairt.gen_ai_api.executors.genie_dialog_factory import GenieDialogFactory
from qairt.gen_ai_api.executors.t2t_executor import T2TExecutor
from qairt.modules.cache_module import CacheModule
from qairt.modules.dlc_module import DlcModule
from qairt.modules.genie_execution.genie_config import BasicDialog, ExportFormat, GenieConfig
from qairt.modules.genie_execution.genie_t2t_runner_helper import GenieT2TRunnerHelper
from qairt.utils import loggers
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DeviceInfo
from qti.aisw.tools.core.utilities.devices.api.device_factory import DeviceFactory
from qti.aisw.tools.core.utilities.devices.api.device_interface import DeviceInterface
from qti.aisw.tools.core.utilities.devices.x86_64_linux.x86_64_linux_device import X86LinuxDevice
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_model_helper import OnnxModelHelper


class ContainerMetadata(AISWBaseModel):
    """
    Container metadata for serialization/deserialization
    """

    backend: BackendType


class LLMContainer(GenAIContainer):
    """
    Produced by an GenAiBuilder and consumed by an GenAiExecutor
    """

    _logger = loggers.get_logger(name=__name__)

    def __init__(
        self,
        models: List[CompiledModel],
        gen_ai_config: GenAIConfig,
        backend: BackendType,
        *,
        backend_extensions_config: Optional[Dict] = None,
        compile_config: Optional[CompileConfig] = None,
    ):
        """
        Create a GenAIContainer

        Args:
            models (CompiledModel): List of CompiledModels containing the LLM split(s) prepared for QAIRT execution
            gen_ai_config (GenAIConfig): contains the configuration metadata for the GenAI model.
            backend (BackendType): The backend the artifacts in this container were prepared for
            backend_extensions_config (Optional[Dict]): Backend extensions configuration for execution
            compile_config (Optional[CompileConfig]): contains configuration metadata for the compiled configuration. Defaults to None.
        """
        if not models:
            raise ValueError("LLMContainer must be have at least one CompiledModel")
        self._models: List[CompiledModel] = models
        self._gen_ai_config: GenAIConfig = gen_ai_config
        self._backend: BackendType = backend
        self._backend_extensions_config: Dict | None = backend_extensions_config

        if compile_config and backend_extensions_config:
            self._logger.warning(
                "Both a CompileConfig and backend extensions config were provided. Ignoring CompileConfig."
            )

        elif isinstance(compile_config, CompileConfig):
            self._backend_extensions_config = compile_config.model_dump(context={"backend_extensions": True})

    @classmethod
    def _metadata_file(cls, path: str | PathLike) -> str:
        return os.path.join(path, "metadata.json")

    @classmethod
    def _gen_ai_config_file(cls, path: str | PathLike) -> str:
        return os.path.join(path, "gen_ai_config.json")

    @classmethod
    def _backend_ext_file(cls, path: str | PathLike) -> str:
        return os.path.join(path, "backend_extensions.json")

    @classmethod
    def _tokenizer_file(cls, path: str | PathLike) -> str:
        return os.path.join(path, "tokenizer.json")

    @classmethod
    def _model_dir(cls, path: str | PathLike) -> str:
        return os.path.join(path, "models")

    @classmethod
    def _create_split_dir(cls, path: Union[str, PathLike], index: int) -> str:
        split_dir = cls._split_dir(path, index)
        os.makedirs(split_dir, exist_ok=True)
        return split_dir

    @classmethod
    def _split_dir(cls, path: Union[str, PathLike], index: int) -> str:
        return os.path.join(cls._model_dir(path), f"split_{index}")

    @classmethod
    def _model_dlc(cls, path: str | PathLike, index: int) -> str:
        return os.path.join(cls._split_dir(path, index), f"model.dlc")

    @classmethod
    def _model_ctx_bin(cls, path: str | PathLike, index: int) -> str:
        return os.path.join(cls._split_dir(path, index), f"model.bin")

    @classmethod
    def _lora_dir(cls, path: str | PathLike) -> str:
        return os.path.join(path, "lora")

    @classmethod
    def _lora_bin(cls, path: str | PathLike, use_case_name: str, index: int) -> str:
        return os.path.join(cls._split_dir(path, index), f"{use_case_name}.bin")

    @classmethod
    def _chat_template_dir(cls, path: str | PathLike) -> str:
        return os.path.join(path, "chat_template")

    @classmethod
    def _embedding_table(cls, path: str | PathLike) -> str:
        return os.path.join(path, "embedding_table.bin")

    @classmethod
    def _draft_container(cls, path: str | PathLike) -> str:
        return os.path.join(path, "eaglet_draft_container")

    @classmethod
    def _eaglet_token_map(cls, path: str | PathLike) -> str:
        return os.path.join(path, "eaglet_token_map.json")

    def write_use_cases_json(self, dest: str | os.PathLike):
        """
        Extracts all unique use case names from the models' lora_use_case_binary_map
        and writes them to a JSON file named 'use_cases.json' inside the models directory.

        Args:
            dest (str | PathLike): Base destination path where the models directory resides.
        """
        use_cases: set[str] = set()
        for model in self._models:
            if hasattr(model, "lora_use_case_binary_map"):
                use_cases.update(model.lora_use_case_binary_map.keys())

        models_dir = os.path.join(dest, "models")
        os.makedirs(models_dir, exist_ok=True)

        use_cases_path = os.path.join(models_dir, "use_cases.json")
        with open(use_cases_path, "w") as f:
            json.dump({"use_cases": sorted(use_cases)}, f, indent=2)

    def save(self, dest: str | PathLike, *, exist_ok: bool = False):
        """
        Save all artifacts to disk.  Note, this will copy artifacts into the destination directory, and update any
        configurations accordingly.

        Args:
            dest (str | PathLike): Path to save the artifacts
        """
        if os.path.exists(dest):
            if not os.path.isdir(dest):
                raise NotADirectoryError(f"Destination path {dest} exists but is not a directory")
            elif not exist_ok:
                raise ValueError(
                    f"Destination path {dest} already exists.  To use an existing directory, specify exist_ok=True"
                )
        else:
            os.makedirs(dest, exist_ok=exist_ok)

        # copy the tokenizer found at self._gen_ai_config.tokenizer_path to dest
        shutil.copyfile(self._gen_ai_config.tokenizer_path, self._tokenizer_file(dest))

        if self._gen_ai_config.chat_template:
            chat_template_dest = self._chat_template_dir(dest)
            self._gen_ai_config.chat_template.save(chat_template_dest)

        # Copy embedding LUT if present
        if self._gen_ai_config.embedding_config:
            if not os.path.exists(self._gen_ai_config.embedding_config.embed_path):
                raise FileNotFoundError(
                    f"Embedding LUT file {self._gen_ai_config.embedding_config.embed_path} does not exist."
                )
            shutil.copyfile(self._gen_ai_config.embedding_config.embed_path, self._embedding_table(dest))

        if self._models:
            os.makedirs(self._model_dir(dest), exist_ok=True)

            for i, model in enumerate(self._models):
                self._create_split_dir(dest, i)

                if isinstance(model.module, CacheModule):
                    model.module.save(self._model_ctx_bin(dest, i))

                    if model.lora_use_case_binary_map:
                        for use_case_name, path in model.lora_use_case_binary_map.items():
                            if use_case_name == "base":
                                continue  # Already saved
                            shutil.copy2(path, self._lora_bin(dest, use_case_name, i))

                elif isinstance(model.module, DlcModule):
                    model.module.save(self._model_dlc(dest, i))
                else:
                    raise TypeError(f"Unsupported Compiled model module type {type(model.module)}")

            self.write_use_cases_json(dest)

        with open(self._metadata_file(dest), "w") as f:
            f.write(ContainerMetadata(backend=self._backend).model_dump_json())

        if isinstance(self._gen_ai_config.speculative_run_config, EagletRunConfig):
            assert self._gen_ai_config.speculative_run_config.draft_container
            shutil.copytree(
                self._gen_ai_config.speculative_run_config.draft_container,
                self._draft_container(dest),
                dirs_exist_ok=True,
            )
            if self._gen_ai_config.speculative_run_config.draft_token_map:
                shutil.copyfile(
                    self._gen_ai_config.speculative_run_config.draft_token_map, self._eaglet_token_map(dest)
                )

        with open(self._gen_ai_config_file(dest), "w") as f:
            f.write(self._gen_ai_config.model_dump_json(by_alias=True, exclude_none=True, indent=2))

        if self._backend_extensions_config:
            with open(self._backend_ext_file(dest), "w") as f:
                f.write(json.dumps(self._backend_extensions_config, indent=2))

    @classmethod
    def _load_lora_binaries(
        cls, path: Union[str, PathLike], index: int, use_cases: List[str], base_filename: str
    ) -> dict:
        """
        Load LoRA binaries from the split directory for a given model index.

        Args:
            path (str | PathLike): Base path to the container.
            index (int): Model index.
            base_filename (str): Name of the base model file to exclude.

        Returns:
            dict: Mapping of use_case_name to Path of LoRA binary.
        """
        lora_map = {}
        split_dir = Path(cls._split_dir(path, index))

        for use_case_name in use_cases:
            if use_case_name == "base":
                continue
            candidate = split_dir / f"{use_case_name}.bin"
            if candidate.exists():
                lora_map[use_case_name] = candidate

        return lora_map

    @classmethod
    def load(cls, path: str | PathLike) -> "LLMContainer":
        """
        Load LLMContainer assets from disk

        Args:
            path (str | PathLike): Path to load the artifacts from.  This should be a directory that is produced
            from a previous call to LLMContainer.save().

        Returns:
            LLMContainer: LLMContainer instance with the loaded artifacts
        """
        if not os.path.isdir(path):
            raise NotADirectoryError(
                f"Path {path} is not a directory.  This should be a directory containing (minimally) a config file "
                f"{cls._gen_ai_config_file('')}, a tokenizer file ({cls._tokenizer_file('')}), and a model directory "
                f"({cls._model_dir('')}). There may be other files and directories as well."
            )

        backend = None
        with open(cls._metadata_file(path), "r") as f:
            backend = ContainerMetadata(**json.load(f)).backend

        gen_ai_config = None
        with open(cls._gen_ai_config_file(path), "r") as f:
            gen_ai_config = GenAIConfig(**json.load(f))

        gen_ai_config.tokenizer_path = str(Path(cls._tokenizer_file(path)).resolve())
        if not os.path.exists(gen_ai_config.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at location: {gen_ai_config.tokenizer_path}")

        if gen_ai_config.embedding_config:
            gen_ai_config.embedding_config.embed_path = str(Path(cls._embedding_table(path)).resolve())
            if not os.path.exists(gen_ai_config.embedding_config.embed_path):
                raise FileNotFoundError(
                    f"Embedding LUT file not found at location: {gen_ai_config.embedding_config.embed_path}"
                )
        if isinstance(gen_ai_config.speculative_run_config, EagletRunConfig):
            gen_ai_config.speculative_run_config.draft_token_map = Path(cls._eaglet_token_map(path)).resolve()
            gen_ai_config.speculative_run_config.draft_container = Path(cls._draft_container(path)).resolve()

        backend_extensions_config = None
        if os.path.exists(cls._backend_ext_file(path)):
            with open(cls._backend_ext_file(path), "r") as f:
                backend_extensions_config = json.load(f)

        models = []
        if not os.path.isdir(cls._model_dir(path)):
            raise NotADirectoryError(f"Serialized model directory does not exist: {cls._model_dir(path)}")

        i = 0
        use_cases_path = os.path.join(cls._model_dir(path), "use_cases.json")
        try:
            with open(use_cases_path, "r") as f:
                use_cases = json.load(f)["use_cases"]
        except FileNotFoundError:
            use_cases = []

        while os.path.isfile(cls._model_ctx_bin(path, i)) or os.path.isfile(cls._model_dlc(path, i)):
            model = None
            lora_use_case_binary_map = {}

            # Load base model
            if os.path.isfile(cls._model_ctx_bin(path, i)):
                base_path = Path(cls._model_ctx_bin(path, i))
                model = CompiledModel(CacheModule.load(path=base_path))
                lora_use_case_binary_map["base"] = base_path

                lora_use_case_binary_map.update(cls._load_lora_binaries(path, i, use_cases, f"model_{i}.bin"))

            elif os.path.isfile(cls._model_dlc(path, i)):
                base_path = Path(cls._model_dlc(path, i))
                model = CompiledModel(DlcModule.load(path=base_path))
                lora_use_case_binary_map["base"] = base_path
                lora_use_case_binary_map.update(cls._load_lora_binaries(path, i, use_cases, f"model_{i}.dlc"))

            if model is not None:
                model.lora_use_case_binary_map = lora_use_case_binary_map
                models.append(model)

            i += 1

        return LLMContainer(
            models, gen_ai_config, backend, backend_extensions_config=backend_extensions_config
        )

    def get_executor(
        self,
        device: Optional[Device] = None,
        clean_up: bool = True,
        prepare_environment: bool = True,
        **kwargs,
    ) -> T2TExecutor:
        """
        Gets a T2TExecutor for the models loaded in this container.

        This function returns an executor that can be used to run inference on the models
        loaded in this container. The executor should either be used in a context manager
        with clean_up = True to automatically remove artifacts upon exit, or the user should
        call clean_environment() explicitly on the returned executor when done to free resources.

        Args:
            device: Optional device to run the executor on. If None, uses the default device.
            clean_up: Whether to clean up resources when the executor is done.
            prepare_environment: Whether to prepare the environment for the executor.
            **kwargs: Additional keyword arguments to pass to the executor.

        Returns:
            A T2TExecutor instance configured with the models in this container.

        Raises:
            ValueError: If no models were loaded into the container.
        """
        if not self._models:
            raise ValueError("No models were loaded into the container. Nothing to execute.")
        draft_container = None
        draft_models = None
        draft_model_backend_extensions_config = None

        if (
            self._gen_ai_config.speculative_run_config
            and isinstance(self._gen_ai_config.speculative_run_config, EagletRunConfig)
            and self._gen_ai_config.speculative_run_config.draft_container
        ):
            draft_container = LLMContainer.load(self._gen_ai_config.speculative_run_config.draft_container)
            draft_models = draft_container._models
            draft_model_backend_extensions_config = draft_container._backend_extensions_config

        executor = T2TExecutor(
            self._models,
            self._gen_ai_config,
            self._backend,
            device=device,
            backend_extensions_config=self._backend_extensions_config,
            qairt_sdk_root=kwargs.get("qairt_sdk_root"),
            clean_up=clean_up,
            draft_models=draft_models,
            draft_model_backend_extensions_config=draft_model_backend_extensions_config,
        )
        if prepare_environment:
            executor.prepare_environment()
        return executor

    def _extract_binary_info_from_cache(self, cache_module: CacheModule, binary_type: str = "target") -> dict:
        raise NotImplementedError("binary-info in the genie metadata isn't currently supported.")

    def _generate_genie_dlc_metadata(self, genie_config: GenieConfig) -> dict:
        """
        Generate Genie DLC metadata conforming to the schema.

        Args:
            genie_config: The GenieConfig object for this container

        Returns:
            dict: Metadata structure conforming to Genie metadata schema
        """

        # Generate header with version 1.0.0
        header = {
            "version": {"major": 1, "minor": 0, "patch": 0},
            "timestamp": {"created": int(time.time()), "last-modified": int(time.time())},
            "createdBy": {"tool": "QAIRT", "version": sdk_version},
        }

        records = []

        # Add context binary records
        dialog_obj = genie_config.dialog
        ctx_bins = []
        if isinstance(dialog_obj, BasicDialog):
            binary = dialog_obj.engine.model.binary
            if binary and getattr(binary, "ctx_bins", None):
                ctx_bins = list(binary.ctx_bins)

        for idx, model in enumerate(self._models):
            if isinstance(model.module, CacheModule) and model.module.path:
                rec_name = Path(str(ctx_bins[idx])).name
                record = {
                    "name": rec_name,
                    "type": "context-binary",
                    "binary-info": None,  # TODO: AISW-171310 Extend Genie metadata support
                }
                records.append(record)

        records.append({"name": "genie_config.json", "type": "genie-config"})

        # Build use-cases array
        use_cases = [{"name": "default", "config": {"type": "node", "name": "genie_config.json"}}]

        return {"header": header, "records": records, "use-cases": use_cases}

    def _export_container_as_dlc(
        self,
        dest: str | PathLike,
        genie_config: GenieConfig,
    ) -> None:
        """
        Export the LLM container as a DLC with all files embedded as records.

        Args:
            dest: Destination path for the output DLC file
            genie_config: Prepared GenieConfig object

        Raises:
            ImportError: If qti.aisw.dlc_utils cannot be imported
        """
        try:
            import qti.aisw.dlc_utils as dlc
        except ImportError:
            raise ImportError("Unable to import qti.aisw.dlc_utils for exporting container as DLC.")

        # Create seed DLC
        seed_dlc_module: DlcModule = self._get_seed_dlc()

        # Generate genie metadata
        genie_metadata = self._generate_genie_dlc_metadata(genie_config)

        # Converts genie_config from dialog format to lm-executor node
        exported_config = genie_config.export(ExportFormat.LM_EXECUTOR)

        # Write artifacts to a temp directory and add as records
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Add genie metadata file to DLC
            metadata_path = os.path.join(tmp_dir, "genie_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(genie_metadata, f, indent=2)

            seed_dlc_module.updater.add_record(metadata_path, dlc.modeltools.DlcRecordType.GENAI_METADATA)

            # Add genie config file to DLC
            config_path = os.path.join(tmp_dir, "genie_config.json")
            with open(config_path, "w") as f:
                json.dump(exported_config, f, indent=2)

            seed_dlc_module.updater.add_record(config_path, dlc.modeltools.DlcRecordType.GENAI_ARTIFACT)

            # Add context-binary records using ctx-bins from the GenieConfig object
            dialog_obj = genie_config.dialog
            if isinstance(dialog_obj, BasicDialog):
                binary = dialog_obj.engine.model.binary
                if binary:
                    for cb in binary.ctx_bins:
                        rec_path = os.path.join(dest, str(cb))
                        if not os.path.isfile(rec_path):
                            raise FileNotFoundError(f"Context binary not found for DLC export: {rec_path}")
                        seed_dlc_module.updater.add_record(
                            rec_path, dlc.modeltools.DlcRecordType.GENAI_ARTIFACT
                        )

            # Save final DLC
            seed_dlc_module.updater.save()
            timestamped_name = f"model-{datetime.now().astimezone().strftime('%Y%m%d-%H%M')}.dlc"
            seed_dlc_module.save(os.path.join(dest, timestamped_name))

    def export(
        self,
        dest: str | PathLike,
        *,
        export_format: ExportFormat = ExportFormat.DIALOG,
        target_device: Optional[Device] = None,
        push_to_target: bool = False,
    ):
        """Export container artifacts to a destination directory on the host or on a target device.

        Container artifacts for the model are always included. If a target_device is specified, then
        bin/lib files from the SDK for that target are included in the export. Further, if push_to_target
        is True, then dest should be a location on the target_device and all exported files will be placed
        at that location.

        The exported contents are identical to what is produced by the ``GenieT2TRunner.load()`` method.

        Args:
            dest: The path where the container contents will be exported. If ``push_to_target`` is
                ``True`` this is a path on the target device.
            export_format: The export format that should be used
            target_device: The target device to collect SDK artifacts for.
            push_to_target: When ``True`` the export is performed directly on the device using the device
                interface; otherwise the export is performed locally.
        Raises:
            NotADirectoryError: If the destination path exists but is not a directory.
        """
        qairt_sdk_root: str | os.PathLike = str(os.environ.get("QAIRT_SDK_ROOT", ""))

        if push_to_target and not target_device:
            raise ValueError("A target device must be specified to push to it.")

        dest_device_interface: DeviceInterface = X86LinuxDevice()
        if target_device:
            target_device_interface: DeviceInterface = (
                target_device.info
                if isinstance(target_device.info, DeviceInterface)
                else DeviceFactory.create_device(target_device.info)
            )
            if push_to_target:
                dest_device_interface = target_device_interface

        dest_device_interface.remove(dest)
        dest_device_interface.make_directory(dest)

        # Target platform libs/bins are only collected if a target platform was explicitly given
        if target_device:
            target_bins, target_libs = GenieT2TRunnerHelper.get_sdk_artifacts(
                target_device_interface, qairt_sdk_root, self._backend
            )

            lib_dir = os.path.join(dest, "lib")
            dest_device_interface.make_directory(lib_dir)
            for lib in target_libs:
                dest_device_interface.push(lib, os.path.join(lib_dir, os.path.basename(lib)))

            bin_dir = os.path.join(dest, "bin")
            dest_device_interface.make_directory(bin_dir)
            for bin in target_bins:
                dest_device_interface.push(bin, os.path.join(bin_dir, os.path.basename(bin)))

        backend_extensions_path = None
        if self._backend_extensions_config:
            with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmp_cfg:
                tmp_cfg.write(json.dumps(self._backend_extensions_config, indent=2))
                backend_extensions_path = tmp_cfg.name

        draft_container_models = None
        draft_container_backend_extensions_path = None
        if (
            self._gen_ai_config.speculative_run_config
            and isinstance(self._gen_ai_config.speculative_run_config, EagletRunConfig)
            and self._gen_ai_config.speculative_run_config.draft_container
        ):
            draft_container = LLMContainer.load(self._gen_ai_config.speculative_run_config.draft_container)
            draft_container_models = draft_container._models if draft_container else None
            if draft_container._backend_extensions_config:
                with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmp_cfg:
                    tmp_cfg.write(json.dumps(draft_container._backend_extensions_config, indent=2))
                    draft_container_backend_extensions_path = tmp_cfg.name

        try:
            dialog = GenieDialogFactory.create_dialog(
                self._backend,
                self._gen_ai_config,
                self._models,
                backend_extensions_path or "",
                draft_models=draft_container_models,
                draft_backend_extensions_path=draft_container_backend_extensions_path or "",
            )
            config = GenieConfig(dialog=dialog)
            _config, _config_artifacts = GenieT2TRunnerHelper.prepare_config(config)

            with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp_cfg:
                tmp_cfg.write(json.dumps(_config.export(export_format), indent=2))
                tmp_cfg_path = tmp_cfg.name
            dest_device_interface.push(tmp_cfg_path, os.path.join(dest, "genie_config.json"))

            # Export artifacts using device interface
            artifacts_dir = os.path.join(dest, "artifacts")
            dest_device_interface.make_directory(artifacts_dir)
            for dst, src in _config_artifacts.items():
                dest_device_interface.push(src, os.path.join(artifacts_dir, os.path.basename(dst)))
        finally:
            if backend_extensions_path:
                os.unlink(backend_extensions_path)
            if draft_container_backend_extensions_path:
                os.unlink(draft_container_backend_extensions_path)
            if tmp_cfg_path and os.path.exists(tmp_cfg_path):
                os.unlink(tmp_cfg_path)

        if export_format == ExportFormat.LM_EXECUTOR:
            # Move backend extensions file out of artifacts directory and update config path
            if isinstance(_config.dialog, BasicDialog):
                extensions_rel_path = getattr(_config.dialog.engine.backend, "extensions", None)
                if extensions_rel_path:
                    ext_name = os.path.basename(extensions_rel_path)
                    src_path = os.path.join(artifacts_dir, ext_name)
                    dst_path = os.path.join(dest, ext_name)
                    # Move via pull/push to support remote devices
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_local:
                        tmp_local_path = tmp_local.name
                    try:
                        dest_device_interface.pull(src_path, tmp_local_path)
                        dest_device_interface.push(tmp_local_path, dst_path)
                        dest_device_interface.remove(src_path)
                        # Update GenieConfig to point to the new top-level path for DLC export
                        _config.dialog.engine.backend.extensions = ext_name
                    finally:
                        if os.path.exists(tmp_local_path):
                            os.unlink(tmp_local_path)

            # Export LLMContainer as DLC
            self._export_container_as_dlc(dest, _config)

            # Remove artifacts directory and genie_config.json after DLC export
            dest_device_interface.remove(artifacts_dir)
            dest_device_interface.remove(os.path.join(dest, "genie_config.json"))

    @staticmethod
    def _get_seed_dlc() -> DlcModule:
        """
        Create a minimal ONNX model (single Identity node) and convert it to a DLC, returning a DlcModule.

        Returns:
            DlcModule: A DLC module produced by converting the generated ONNX model.

        Raises:
            ImportError: If converter dependencies are unavailable (Convert is not supported on OE-Linux).
            onnx.checker.ValidationError: If the generated ONNX model fails ONNX validation.
            qairt.utils.exceptions.ConversionError: If ONNX-to-DLC conversion fails.
        """
        try:
            from qairt.api.converter._convert import convert  # lazy import: build-time only
        except ImportError as e:
            raise ImportError("Converter dependencies are not available.") from e

        # Create minimal ONNX model via ONNX utils
        model = OnnxModelHelper.create_minimal_onnx_model()

        # Serialize to a temporary .onnx file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp_onnx:
            OnnxModelHelper.save_model(model, tmp_onnx.name)
            onnx_path = tmp_onnx.name

        # Convert ONNX to DLC using QAIRT converter
        try:
            converted_model = convert(onnx_path)
        finally:
            # Clean up the temporary ONNX file
            os.unlink(onnx_path)

        # Return the DlcModule
        return converted_model.module
