# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import json
import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator, validator
from typing_extensions import Self

from qairt.api.configs.common import AISWBaseModel


class GenieConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, os.PathLike):
            return str(obj)
        return super().default(obj)


class VersionedModel(AISWBaseModel):
    version: int = 1
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)


class QnnHtpBackend(VersionedModel):
    use_mmap: bool = Field(False, alias="use-mmap")
    spill_fill_bufsize: int = Field(0, alias="spill-fill-bufsize")
    mmap_budget: int = Field(40, alias="mmap-budget")
    poll: bool
    pos_id_dim: Optional[int] = Field(None, alias="pos-id-dim")
    cpu_mask: str = Field(default="0x00", alias="cpu-mask")
    kv_dim: Optional[int] = Field(None, alias="kv-dim")
    kv_update_method: Optional[str] = Field(None, alias="kv-update-method")
    rope_theta: Optional[int] = Field(None, alias="rope-theta")
    allow_async_init: Optional[bool] = Field(None, alias="allow-async-init")
    enable_graph_switching: Optional[bool] = Field(None, alias="enable-graph-switching")


class QnnGenAiTransformerBackend(VersionedModel):
    use_mmap: Optional[bool] = Field(None, alias="use-mmap")
    n_logits: Optional[int] = Field(None, alias="n-logits")
    n_layer: Optional[int] = Field(None, alias="n-layer")
    n_embd: Optional[int] = Field(None, alias="n-embd")
    n_heads: Optional[int] = Field(None, alias="n-heads")


class EngineBackendType(str, Enum):
    QNN_GEN_AI_TRANSFORMER = "QnnGenAiTransformer"
    QNN_HTP = "QnnHtp"


class EngineBackend(VersionedModel):
    type: EngineBackendType = EngineBackendType.QNN_GEN_AI_TRANSFORMER
    QnnGenAiTransformer: Optional[QnnGenAiTransformerBackend] = None
    QnnHtp: Optional[QnnHtpBackend] = None
    extensions: Optional[str | os.PathLike] = None

    @model_validator(mode="after")
    def check_type(self) -> Self:
        if self.type == EngineBackendType.QNN_GEN_AI_TRANSFORMER:
            if self.QnnGenAiTransformer is None:
                raise ValueError(f"QnnGenAiTransformer must be provided when type is: {self.type.value}")
        elif self.QnnGenAiTransformer is not None:
            raise ValueError(
                "QnnGenAiTransformer should only be provided when type is: "
                f"{EngineBackendType.QNN_GEN_AI_TRANSFORMER.value}"
            )

        if self.type == EngineBackendType.QNN_HTP:
            if self.QnnHtp is None:
                raise ValueError(f"QnnHtp must be provided when type is: {self.type.value}")
        elif self.QnnHtp is not None:
            raise ValueError(
                f"QnnHtp should only be provided when type is: {EngineBackendType.QNN_HTP.value}"
            )

        return self


class LoraConfigAdapter(VersionedModel):
    name: str
    alphas: List[str] = Field(default_factory=list)
    bin_sections: List[str | os.PathLike] = Field(default_factory=list, alias="bin-sections")


class LoraConfig(VersionedModel):
    alpha_tensor_name: Optional[str] = Field(None, alias="alpha-tensor-name")
    adapters: List[LoraConfigAdapter]


class ModelBinary(VersionedModel):
    ctx_bins: List[str | os.PathLike] = Field(alias="ctx-bins")
    lora: Optional[LoraConfig] = None


class ModelLibrary(VersionedModel):
    model_bin: str | os.PathLike = Field(alias="model-bin")
    lora: Optional[LoraConfig] = None


class RopeType(str, Enum):
    LLAMA3 = "llama3"
    DEFAULT = "default"
    LONG_ROPE = "longrope"


class RopeScaling(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    rope_type: Optional[RopeType] = Field(None, alias="rope-type")
    factor: Optional[float] = None
    low_freq_factor: Optional[float] = Field(None, alias="low-freq-factor")
    high_freq_factor: Optional[float] = Field(None, alias="high-freq-factor")
    original_max_position_embeddings: Optional[int] = Field(None, alias="original-max-position-embeddings")
    short_factor: Optional[List[float]] = Field(None, alias="short-factor")
    long_factor: Optional[List[float]] = Field(None, alias="long-factor")


class PositionalEncodingType(str, Enum):
    ROPE = "rope"
    ABSOLUTE = "absolute"
    ALIBI = "alibi"


class PositionalEncoding(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Optional[PositionalEncodingType] = None
    rope_dim: Optional[int] = Field(None, alias="rope-dim")
    rope_theta: Optional[float] = Field(None, alias="rope-theta")
    rope_scaling: Optional[RopeScaling] = Field(None, alias="rope-scaling")


class EngineModelType(str, Enum):
    LIBRARY = "library"
    BINARY = "binary"


class EngineModel(VersionedModel):
    type: EngineModelType = EngineModelType.LIBRARY
    library: Optional[ModelLibrary] = None
    binary: Optional[ModelBinary] = None
    positional_encoding: Optional[PositionalEncoding] = Field(None, alias="positional-encoding")


class DraftEngineModel(EngineModel):
    draft_token_map: Optional[str | os.PathLike] = Field(None, alias="draft-token-map")

    @validator("draft_token_map", pre=True)
    def parse_draft_token_map(cls, v):
        if v == "None" or v == "":
            return None
        return v


class DialogEngine(VersionedModel):
    n_threads: int = Field(6, alias="n-threads")
    backend: EngineBackend = Field(default_factory=EngineBackend)
    model: EngineModel = Field(default_factory=EngineModel)


class EagletEngineRole(str, Enum):
    TARGET = "target"
    DRAFT = "draft"


class EagletDraftDialogEngine(DialogEngine):
    role: Literal[EagletEngineRole.DRAFT] = EagletEngineRole.DRAFT
    model: DraftEngineModel = Field(default_factory=DraftEngineModel)


class EagletTargetDialogEngine(DialogEngine):
    role: Literal[EagletEngineRole.TARGET] = EagletEngineRole.TARGET


class Context(VersionedModel):
    bos_token: int = Field(0, alias="bos-token")
    eos_token: int | List[int] = Field(0, alias="eos-token")
    eot_token: Optional[int] = Field(None, alias="eot-token")
    n_vocab: int = Field(0, alias="n-vocab")
    size: int = 512
    pad_token: Optional[int] = Field(None, alias="pad-token")


class EagletContext(Context):
    draft_n_vocab: Optional[int] = Field(alias="draft-n-vocab")


class Sampler(VersionedModel):
    seed: int = 42
    temp: float = 0.8
    top_k: int = Field(40, alias="top-k")
    top_p: float = Field(0.95, alias="top-p")
    greedy: Optional[bool] = None
    type: Optional[str] = None
    callback_name: Optional[str] = Field(None, alias="callback-name")


class Tokenizer(VersionedModel):
    path: str | os.PathLike = ""


class DialogEmbeddingDataType(str, Enum):
    FLOAT32 = "float32"
    NATIVE = "native"
    UFIXED8 = "ufixed8"
    UFIXED16 = "ufixed16"


class DialogEmbeddingQuantParam(VersionedModel):
    scale: float
    offset: int


class DialogEmbedding(VersionedModel):
    type: str = "lut"
    lut_path: str | os.PathLike = Field(alias="lut-path")
    size: int
    datatype: DialogEmbeddingDataType
    quant_param: Optional[DialogEmbeddingQuantParam] = Field(None, alias="quant-param")


class SsdConfig(VersionedModel):
    ssd_version: int = Field(1, alias="ssd-version")
    forecast_token_count: int = Field(alias="forecast-token-count")
    forecast_prefix: int = Field(alias="forecast-prefix")
    branches: List[int]
    n_streams: Optional[int] = Field(None, alias="n-streams")
    p_threshold: Optional[float] = Field(None, alias="p-threshold")


class SsdRunConfig(SsdConfig):
    """Configuration for running SSD inference."""

    forecast_prefix_name: str | os.PathLike = Field(alias="forecast-prefix-name")


class LADEType(str, Enum):
    ALWAYS_FWD_ONE = "ALWAYS_FWD_ONE"
    FWD_MAX_HIT = "FWD_MAX_HIT"
    FWD_LEVEL = "FWD_LEVEL"


class LadeConfig(VersionedModel):
    update_mode: LADEType = Field(LADEType.ALWAYS_FWD_ONE, alias="update-mode")
    window: int
    ngram: int
    gcap: int


class EagletConfig(VersionedModel):
    eaglet_version: int = Field(1, alias="eaglet-version")
    draft_len: int = Field(alias="draft-len")
    n_branches: int = Field(alias="n-branches")
    max_tokens_target_can_evaluate: int = Field(alias="max-tokens-target-can-evaluate")
    draft_kv_cache: bool = Field(alias="draft-kv-cache")


class DialogType(str, Enum):
    BASIC = "basic"
    SSD_Q1 = "ssd-q1"
    LADE = "lade"
    EAGLET = "eaglet"


class AbstractDialog(VersionedModel):
    type: str
    tokenizer: Tokenizer
    stop_sequence: Optional[List[str]] = Field(None, alias="stop-sequence")
    max_num_tokens: Optional[int] = Field(None, alias="max-num-tokens")
    sampler: Optional[Sampler] = None
    embedding: Optional[DialogEmbedding] = None
    context: Context


class BasicDialog(AbstractDialog):
    type: Literal["basic"] = "basic"
    engine: DialogEngine


class SSDDialog(AbstractDialog):
    type: Literal["ssd-q1"] = "ssd-q1"
    ssd_q1: SsdRunConfig = Field(alias="ssd-q1")
    engine: DialogEngine


class LadeDialog(AbstractDialog):
    type: Literal["lade"] = "lade"
    lade: LadeConfig
    engine: DialogEngine


class EagletDialog(AbstractDialog):
    type: Literal["eaglet"] = "eaglet"
    eaglet: EagletConfig
    context: EagletContext
    engine: List[EagletDraftDialogEngine | EagletTargetDialogEngine]


class ExportFormat(Enum):
    """
    Container export formats
    """

    DIALOG = "dialog"
    LM_EXECUTOR = "lm_executor"


class GenieConfig(AISWBaseModel):
    """
    top level config object for genie config
    """

    dialog: Union[BasicDialog, LadeDialog, SSDDialog, EagletDialog] = Field(discriminator="type")

    def export(self, export_format: ExportFormat = ExportFormat.DIALOG) -> dict[str, Any]:
        model_as_dict = self.model_dump(by_alias=True, exclude_none=True)
        # Converts GenieConfig from dialog format to lm-executor node
        if export_format == ExportFormat.LM_EXECUTOR:
            # Rename top-level key from dialog to lm-executor node
            model_as_dict["lm-executor"] = model_as_dict["dialog"]
            del model_as_dict["dialog"]

            # Remove irrelevant top-level keys for lm-executor node
            del model_as_dict["lm-executor"]["type"]
            del model_as_dict["lm-executor"]["tokenizer"]
            del model_as_dict["lm-executor"]["sampler"]

            # Add configurations specific for lm-executor node
            model_as_dict["lm-executor"]["buffer-type"] = "persistent"
            model_as_dict["lm-executor"]["execution-strategy"] = "explicit"

            # Rewrite ctx-bins to use record:// URIs for DLC export
            if isinstance(self.dialog, BasicDialog):
                binary = model_as_dict["lm-executor"]["engine"]["model"]["binary"]
                binary["ctx-bins"] = [
                    f"record://{os.path.basename(str(path))}" for path in binary["ctx-bins"]
                ]

        return model_as_dict

    def __str__(self):
        return json.dumps(self.export(), indent=2, cls=GenieConfigEncoder)
