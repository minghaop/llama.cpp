# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
Auto configuration and model factory classes for supported architectures.

This module provides factory classes that automatically instantiate the correct
configuration and model classes based on the architecture type specified in the
model configuration.
"""

from qti.aisw.graph_gen.architectures.llama.config import LLaMaConfig, QwenConfig, Phi3Config, MistralConfig
from qti.aisw.graph_gen.architectures.llama.model import LLaMa

# Mapping of architecture names to their corresponding configuration classes
_SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM": LLaMaConfig,
    "Qwen2ForCausalLM": QwenConfig,
    "Phi3ForCausalLM": Phi3Config,
    "MistralForCausalLM": MistralConfig
}


# Mapping of configuration classes to their corresponding model classes
_CONFIG_TO_MODEL = {
    LLaMaConfig: LLaMa,
    QwenConfig: LLaMa,
    Phi3Config: LLaMa,
    MistralConfig: LLaMa,
}
_SUPPORTED_CONFIG_TYPES = tuple(_CONFIG_TO_MODEL.keys())


class AutoConfig:
    """
    Factory class for automatically creating configuration objects based on architecture type.
    
    This class uses the factory pattern via __new__ to return instances of specific
    configuration classes (e.g., LLaMaConfig, QwenConfig) based on the 'architectures'
    field provided in the kwargs.
    
    Usage:
        config = AutoConfig(architectures=["LlamaForCausalLM"], **other_config_params)
        # Returns an instance of LLaMaConfig
    
    Raises:
        ValueError: If 'architectures' field is not present in kwargs.
        NotImplementedError: If no supported architecture is found in the provided list.
    """
    
    def __new__(cls, *args, **kwargs):
        """
        Create and return an instance of the appropriate configuration class.
        
        Args:
            *args: Positional arguments to pass to the configuration class constructor.
            **kwargs: Keyword arguments that must include 'architectures' field.
                     All kwargs are passed to the configuration class constructor.
        
        Returns:
            An instance of the appropriate configuration class (e.g., LLaMaConfig, QwenConfig).
        
        Raises:
            ValueError: If 'architectures' field is not present in kwargs.
            NotImplementedError: If none of the architectures in the list are supported.
        """
        if "architectures" not in kwargs:
            raise ValueError("To use AutoConfig, `architectures` field should be present in the config")

        architectures = kwargs['architectures']

        # Validate that architectures is iterable
        if not isinstance(architectures, (list, tuple)):
            raise ValueError(
                f"The 'architectures' field must be a list or tuple, got {type(architectures).__name__}"
            )
        
        # Iterate through the provided architectures and return the first supported one
        for arch in architectures:
            if arch in _SUPPORTED_ARCHITECTURES:
                return _SUPPORTED_ARCHITECTURES[arch](*args, **kwargs)

        raise NotImplementedError(
            f"No supported architecture found for the given config. "
            f"Provided architectures: {architectures}. "
            f"Supported architectures: {list(_SUPPORTED_ARCHITECTURES.keys())}"
        )


class AutoModel:
    """
    Factory class for automatically creating model objects based on configuration type.
    
    This class uses the factory pattern via __new__ to return instances of specific
    model classes based on the type of configuration object provided.
    
    Usage:
        config = LLaMaConfig(...)
        model = AutoModel(config)
        # Returns an instance of LLaMa model
    
    Raises:
        ValueError: If the provided config type is not supported.
    """
    
    def __new__(cls, config):
        """
        Create and return an instance of the appropriate model class.
        
        Args:
            config: A configuration object (e.g., LLaMaConfig, QwenConfig).
        
        Returns:
            An instance of the appropriate model class (e.g., LLaMa).
        
        Raises:
            ValueError: If the config type is not in the supported configurations mapping.
        """
        # Check if the config is an instance of any supported configuration class
        if isinstance(config, _SUPPORTED_CONFIG_TYPES):
            return _CONFIG_TO_MODEL[type(config)](config)
        
        raise ValueError(
            f"Unsupported config of type {type(config)} for AutoModel. "
            f"Supported config types: {list(_CONFIG_TO_MODEL.keys())}"
        )
