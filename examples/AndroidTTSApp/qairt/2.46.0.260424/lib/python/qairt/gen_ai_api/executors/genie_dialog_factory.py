# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
import os
from collections import defaultdict
from typing import DefaultDict, List, Optional

from qairt import CompiledModel
from qairt.api.configs.common import BackendType
from qairt.gen_ai_api.configs.eaglet_config import EagletRunConfig
from qairt.gen_ai_api.configs.gen_ai_config import GenAIConfig
from qairt.gen_ai_api.configs.lade_config import LadeRunConfig
from qairt.modules.cache_module import CacheModule
from qairt.modules.dlc_module.dlc_module import DlcModule
from qairt.modules.genie_execution.genie_config import (
    AbstractDialog,
    BasicDialog,
    Context,
    DialogEmbedding,
    DialogEmbeddingDataType,
    DialogEmbeddingQuantParam,
    DialogEngine,
    DraftEngineModel,
    EagletContext,
    EagletDialog,
    EagletDraftDialogEngine,
    EagletTargetDialogEngine,
    EngineBackend,
    EngineBackendType,
    EngineModel,
    EngineModelType,
    LadeDialog,
    LoraConfig,
    LoraConfigAdapter,
    ModelBinary,
    ModelLibrary,
    QnnGenAiTransformerBackend,
    QnnHtpBackend,
    Sampler,
    SSDDialog,
    SsdRunConfig,
    Tokenizer,
)


class GenieDialogFactory:
    @staticmethod
    def _create_engine_components(
        backend: BackendType,
        config: GenAIConfig,
        models: List[CompiledModel],
        is_native_execution: bool,
        backend_extensions_path=None,
    ) -> DialogEngine:
        model_paths: List[str] = []
        for model in models:
            if isinstance(model.module, DlcModule):
                caches = list(model.module.caches.values())
                if len(caches) > 1:
                    raise ValueError("Expected only a single context binary cache per dlc")
                model_paths.append(str(caches[0].path))

            elif isinstance(model.module, CacheModule):
                model_paths.append(str(model.module.path))
        lora_config = None
        if GenieDialogFactory._should_collect_lora(models):
            lora_use_cases = GenieDialogFactory._collect_lora_use_cases(models)
            lora_config = GenieDialogFactory._build_lora_config(config, lora_use_cases)

        if backend == BackendType.HTP:
            engine_backend = EngineBackend(
                type=EngineBackendType.QNN_HTP,
                QnnHtp=QnnHtpBackend(
                    **{
                        "poll": True,
                        "use-mmap": not is_native_execution,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 40,
                        "kv-dim": config.kv_dim,
                    }
                ),
            )
            if backend_extensions_path:
                engine_backend.extensions = backend_extensions_path
            engine_model = EngineModel(
                type=EngineModelType.BINARY,
                binary=ModelBinary(ctx_bins=model_paths, lora=lora_config),
                positional_encoding=config.positional_encoding,
            )
            return DialogEngine(backend=engine_backend, model=engine_model)

        elif backend == BackendType.CPU:
            engine_backend = EngineBackend(
                type=EngineBackendType.QNN_GEN_AI_TRANSFORMER,
                QnnGenAiTransformer=QnnGenAiTransformerBackend(
                    n_layer=config.n_layer,
                    n_embd=config.n_embd,
                    n_heads=config.n_heads,
                ),
            )
            engine_model = EngineModel(
                type=EngineModelType.LIBRARY, library=ModelLibrary(model_bin=model_paths[0], lora=lora_config)
            )
            return DialogEngine(backend=engine_backend, model=engine_model)

        else:
            raise RuntimeError(f"Unsupported backend: {backend}")

    @staticmethod
    def _get_dialog_embedding(config: GenAIConfig):
        if embedding_config := config.embedding_config:
            data_type = DialogEmbeddingDataType(embedding_config.embed_datatype)
            quant_param = None
            if embedding_config.embed_quant_scale and embedding_config.embed_quant_offset:
                quant_param = DialogEmbeddingQuantParam(
                    scale=embedding_config.embed_quant_scale,
                    offset=embedding_config.embed_quant_offset,
                )
            dialog_embedding = DialogEmbedding(
                type="lut",
                lut_path=embedding_config.embed_path,
                size=embedding_config.embed_length,
                datatype=data_type,
                quant_param=quant_param,
            )
            return dialog_embedding
        return None

    @staticmethod
    def _collect_lora_use_cases(
        models: List[CompiledModel],
    ) -> dict[str, List[str]]:
        """
        Collect LoRA use‑case binary paths from the provided models.

        Returns a dict mapping each use‑case name to a list of binary paths
        (empty string when a model does not provide that use‑case).
        """
        lora_use_cases: DefaultDict[str, List[str]] = defaultdict(list)
        all_unique_uc_names = set()
        for m in models:
            if m.lora_use_case_binary_map:
                for uc_name in m.lora_use_case_binary_map.keys():
                    if uc_name != "base":
                        all_unique_uc_names.add(uc_name)

        for model in models:
            for uc_name in all_unique_uc_names:
                value = model.lora_use_case_binary_map.get(uc_name, "")
                lora_use_cases[uc_name].append(str(value))
        return dict(lora_use_cases)

    @staticmethod
    def _should_collect_lora(models: List[CompiledModel]) -> bool:
        """
        Determine whether any model contains LoRA use‑case information.
        """
        return any(m.lora_use_case_binary_map for m in models if isinstance(m.module, CacheModule))

    @staticmethod
    def _build_lora_config(
        config: GenAIConfig,
        lora_use_cases: dict[str, List[str]],
    ) -> LoraConfig | None:
        """
        Build a :class:`LoraConfig` from the collected ``lora_use_cases``.
        Returns ``None`` when ``lora_use_cases`` is empty.
        """
        if not lora_use_cases:
            return None

        lora_config_adapters = []
        for uc_name, bin_paths in lora_use_cases.items():
            # Determine alphas for this use‑case
            if (adapter_count_dict := config.adapter_count_by_use_case) is not None:
                if uc_name == "default_adapter":
                    alphas = []
                elif adapter_count_dict.get(uc_name) is not None:
                    alphas = [f"alpha{i}" for i in range(adapter_count_dict[uc_name])]
                else:
                    raise KeyError(
                        f"The use case {uc_name} cannot be found in the adapter_count_by_use_case dict."
                    )
                lora_config_adapters.append(
                    LoraConfigAdapter(name=uc_name, alphas=alphas, bin_sections=bin_paths)
                )

        return LoraConfig(
            alpha_tensor_name=config.alpha_tensor_name,
            adapters=lora_config_adapters,
        )

    @staticmethod
    def create_dialog(
        backend: BackendType,
        config: GenAIConfig,
        models: List[CompiledModel],
        backend_extensions_path: Optional[str | os.PathLike] = None,
        is_native_execution: bool = False,
        draft_models: Optional[List[CompiledModel]] = None,
        draft_backend_extensions_path: Optional[str | os.PathLike] = None,
    ) -> AbstractDialog:
        if not models:
            raise RuntimeError("No models were loaded into the container. Cannot build Genie config.")

        sampler = Sampler()
        model_sampler = getattr(config, "sampler_params", None)
        if model_sampler:
            for param in ["seed", "temp", "top_k", "top_p"]:
                if hasattr(model_sampler, param) and getattr(model_sampler, param) is not None:
                    setattr(sampler, param, getattr(model_sampler, param))
        common_params = {
            "context": Context(
                size=config.context_length,
                n_vocab=config.n_vocab,
                bos_token=config.bos_token,
                eos_token=config.eos_token,
                eot_token=config.eot_token,
            ),
            "sampler": sampler,
            "tokenizer": Tokenizer(path=config.tokenizer_path),
            "engine": GenieDialogFactory._create_engine_components(
                backend, config, models, is_native_execution, backend_extensions_path
            ),
            "embedding": GenieDialogFactory._get_dialog_embedding(config),
        }

        if not config.speculative_run_config:
            return BasicDialog(**common_params)
        if isinstance(config.speculative_run_config, LadeRunConfig):
            return LadeDialog(lade=config.speculative_run_config, **common_params)
        if isinstance(config.speculative_run_config, SsdRunConfig):
            return SSDDialog(ssd_q1=config.speculative_run_config, **common_params)
        if isinstance(config.speculative_run_config, EagletRunConfig):
            if not draft_models:
                raise RuntimeError(
                    "No draft models were loaded into the container. Cannot build Genie config."
                )
            n_vocab = None
            if config.speculative_run_config.draft_token_map:
                with open(config.speculative_run_config.draft_token_map, "r") as f:
                    token_map = json.load(f)
                    n_vocab = len(token_map)

            context = common_params["context"]
            common_params["context"] = EagletContext(**context.model_dump(), draft_n_vocab=n_vocab)
            target_engine = common_params["engine"]
            eaglet_target_engine = EagletTargetDialogEngine(**target_engine.model_dump())
            draft_engine = GenieDialogFactory._create_engine_components(
                backend, config, draft_models, is_native_execution, draft_backend_extensions_path
            )
            engine_model = draft_engine.model
            draft_engine_model = DraftEngineModel(
                **engine_model.model_dump(),
                draft_token_map=str(config.speculative_run_config.draft_token_map),
            )

            eaglet_draft_engine = EagletDraftDialogEngine(
                n_threads=draft_engine.n_threads, backend=draft_engine.backend, model=draft_engine_model
            )
            common_params["engine"] = [eaglet_target_engine, eaglet_draft_engine]
            return EagletDialog(eaglet=config.speculative_run_config, **common_params)
        raise NotImplementedError(f"Unknown Speculative Decoding type: {config.speculative_run_config}")
