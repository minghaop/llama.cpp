# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import pathlib
from os import PathLike
from typing import TYPE_CHECKING, Dict, Optional, Union

from pydantic import Field, field_validator

from qairt.api.configs.common import AISWBaseModel
from qairt.gen_ai_api.chat.chat_templates import CustomChatTemplate, HFChatTemplate, NullChatTemplate
from qairt.gen_ai_api.configs.eaglet_config import EagletRunConfig
from qairt.gen_ai_api.configs.lade_config import LadeRunConfig
from qairt.modules.genie_execution.genie_config import PositionalEncoding, RopeScaling, Sampler, SsdRunConfig

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


class EmbeddingConfig(AISWBaseModel):
    """
    EmbeddingConfig holds configuration information for the embedding table LUT.
    """

    embed_path: str | PathLike
    """
    Path to embedding table LUT.
    """

    embed_datatype: str
    """
    Embedding datatype.
    """

    embed_length: int
    """
    Embedding length.
    """

    embed_quant_scale: Optional[float] = None
    """
    Embedding quant scale.
    """
    embed_quant_offset: Optional[int] = None
    """
    Embedding quant offset.
    """


class ExpertConfig(AISWBaseModel):
    """Configuration for Mixture-of-Experts inference behaviour."""

    enable_op_predication: bool = False
    """Enable operation predication for expert routing."""

    enable_expert_subselection: bool = False
    """Enable expert subselection optimisation."""

    @classmethod
    def from_pretrained_config(cls, config: "PretrainedConfig") -> "ExpertConfig":
        """
        Create an ExpertConfig from a HuggingFace PretrainedConfig.
        Fields default to False when absent from the HF config.
        """
        return cls(
            enable_op_predication=getattr(config, "do_op_predication", False),
            enable_expert_subselection=getattr(config, "do_expert_subselection", False),
        )


class GenAIConfig(AISWBaseModel):
    """
    GenAIConfig holds common configuration information for the Generative AI Model, needed for Genie
    execution.  Common attributes (present in all subclasses):
    """

    model_config = {"arbitrary_types_allowed": True}

    tokenizer_path: str | PathLike
    """
    The path to the tokenizer.  Must point to an existing file.
    """
    context_length: int
    """context length"""

    n_vocab: int
    """The number of tokens in the vocabulary, which is also the first dimension of the embeddings matrix"""

    n_heads: Optional[int] = None
    """The number of attention heads used in the multi-head attention layers of the model"""

    n_layer: Optional[int] = None
    """The number of blocks in the model"""

    n_embd: Optional[int] = None
    """The hidden size of the model"""

    bos_token: int
    """The id of the beginning of stream token."""

    eos_token: int | list[int]
    """The id of the end of stream token."""

    eot_token: Optional[int] = None
    """The id of the end of turn token."""

    positional_encoding: Optional[PositionalEncoding] = None
    """An object describing the positional encodings"""

    kv_dim: Optional[int] = None
    """dimension of the kv cache"""

    rope_theta: Optional[float] = None
    """theta value for rotational positional encoding"""

    rope_scaling: Optional[RopeScaling] = None
    """rope scaling configuration for extended context"""

    alpha_tensor_name: Optional[str] = ""
    """
    Name of the tensor where LoRA adapter is being applied.
    """

    adapter_count_by_use_case: Optional[Dict[str, int]] = {}
    """
    Dict of number of adapters per use case.
    """

    embedding_config: Optional[EmbeddingConfig] = None
    """
    Embedding config.
    """
    speculative_run_config: Optional[LadeRunConfig | EagletRunConfig | SsdRunConfig] = None
    """
    Speculative decoding config.
    """

    chat_template: Union[NullChatTemplate, HFChatTemplate, CustomChatTemplate] = Field(
        default_factory=NullChatTemplate, discriminator="type"
    )
    """
    Chat template for message formatting.
    Supports NullChatTemplate, HFChatTemplate, or CustomChatTemplate instances.
    """

    sampler_params: Optional[Sampler] = None
    """
    Model-specific sampler parameters for optimal model performance.
    Applied automatically by the executor.
    """

    expert_config: Optional[ExpertConfig] = None
    """MoE expert configuration. Populated automatically for MoE architectures."""

    @field_validator("tokenizer_path")
    def validate_tokenizer_path(cls, v):
        path = pathlib.Path(v)
        if not path.resolve().is_file():
            raise FileNotFoundError(f"The tokenizer_path '{v}' does not point to an existing file.")
        return v
