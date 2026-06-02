# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
""" File contains the input config for the LLaMa Style LLM architecture """
import logging
from typing import Optional

from aenum import Enum
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class RopeType(str, Enum):
    """
    Defines enumeration for the different rope types.
    """

    LLAMA3 = "llama3"
    DEFAULT = "default"
    LONG_ROPE = "longrope"


class RopeScaling(BaseModel):
    """
    Configuration for rope scaling field.
    """

    model_config = {"extra": "ignore"}
    rope_type: RopeType = None
    factor: float = None
    low_freq_factor: float = None
    high_freq_factor: float = None
    original_max_position_embeddings: float = None
    short_factor: list[float] = None
    long_factor: list[float] = None


class LLaMaConfig(BaseModel):
    """
    Configuration class for LLaMa-like models.
    """

    num_hidden_layers: int
    max_position_embeddings: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    bos_token_id: int
    eos_token_id: int

    # Architecture Specific, defaults value for LLaMa is given
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000.0 # Base used for LLaMa 2
    rope_scaling: Optional[RopeScaling] = None

    # Output model specific
    batch: int = Field(default=1, gt=0)
    seq_length: int = Field(default=1, gt=0)
    sha: bool = True
    split: int = Field(default=1, gt=0)


    @model_validator(mode='after')
    def validate_dimensions(self):
        """
        Validates the dimensional consistency of the LLaMa configuration.

        :raises ValueError: If any of the validation checks fail.
        """

        # Verify hidden_size is divisible by num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        # Verify KV heads is valid
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) cannot exceed "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        return self

    @property
    def head_dim(self) -> int:
        """Computed property - head dimension is derived from hidden_size and attention heads"""
        return self.hidden_size // self.num_attention_heads


class MistralConfig(LLaMaConfig):
    """
    Ministral / Mistral-family configuration where head_dim
    is explicitly specified and may differ from
    hidden_size // num_attention_heads.
    """

    # Store user-provided value using a separate internal name
    head_dim_: Optional[int] = Field(
        default=None,
        alias="head_dim",
        description="Attention head dimension for Ministral models"
    )

    @property
    def head_dim(self) -> int:
        """
        Resolution order:
        1. Use head_dim if provided
        2. Fall back to standard LLaMA behavior
        """
        if self.head_dim_ is not None:
            return self.head_dim_

        return super().head_dim

class QwenConfig(LLaMaConfig):
    """
    Configuration class for Qwen-like models, inheriting from LLaMaConfig.

    Qwen models use a similar architecture to LLaMa but typically include
    attention bias by default.
    """

    # QWen uses the same architecture style as LLaMa with attention bias
    attention_bias: bool = True


class Phi3Config(LLaMaConfig):
    """
    Configuration class for Phi3 models, inheriting from LLaMaConfig.

    Phi3 models use a similar architecture to LLaMa.
    """

    PHI_EOS_TOKEN_ID: int = 32007
    eos_token_id : int | list

    @model_validator(mode='after')
    def update_eos_tokens(self):
        """
        Update the EOS token ID for Phi Model, Required to improve the output quality.
        """

        # Add additional EOS token ID for Phi models
        if isinstance(self.eos_token_id, int) and self.eos_token_id != self.PHI_EOS_TOKEN_ID:
            logger.debug(
                f"Changing EOS token from {self.eos_token_id} to [{self.eos_token_id}, {self.PHI_EOS_TOKEN_ID}] for Phi model"
            )
            self.eos_token_id = [self.eos_token_id, self.PHI_EOS_TOKEN_ID]
        elif isinstance(self.eos_token_id, list):
            if self.PHI_EOS_TOKEN_ID not in self.eos_token_id:
                logger.debug(
                    f"Adding EOS token {self.PHI_EOS_TOKEN_ID} to existing list {self.eos_token_id} for Phi model"
                )
                self.eos_token_id.append(self.PHI_EOS_TOKEN_ID)

        return self
