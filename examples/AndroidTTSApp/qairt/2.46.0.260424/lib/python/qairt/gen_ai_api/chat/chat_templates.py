# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import hashlib
import json
import os
import shutil
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from jinja2 import Environment
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import AutoTokenizer

from qairt.utils.loggers import get_logger


def cache_hf_tokenizer(cache_dir_key: str = "cache_dir"):
    """
    Decorator to cache HuggingFace tokenizer downloads.

    Args:
        cache_dir_key: Name of the keyword argument that contains the cache directory
    """

    def decorator(func):
        @wraps(func)
        def wrapper(cls, pretrained_model_name_or_path: str, kwargs: Optional[Dict] = None):
            if kwargs is None:
                kwargs = {}

            cache_dir = kwargs.get(cache_dir_key)

            if not cache_dir:
                cache_dir = os.getcwd()

            if os.path.exists(pretrained_model_name_or_path):
                return func(cls, pretrained_model_name_or_path, kwargs)

            hf_model_id_safe = pretrained_model_name_or_path.replace("/", "_").replace(":", "_")
            chat_template_cache_dir = os.path.join(cache_dir, "chat_template", hf_model_id_safe)

            if os.path.isdir(chat_template_cache_dir) and os.listdir(chat_template_cache_dir):
                _logger.info(f"Loading cached tokenizer from {chat_template_cache_dir}")
                kwargs_copy = kwargs.copy()
                kwargs_copy.pop(cache_dir_key, None)

                return func(cls, chat_template_cache_dir, kwargs_copy)
            else:
                _logger.info(f"Downloading and caching tokenizer for {pretrained_model_name_or_path}")
                os.makedirs(chat_template_cache_dir, exist_ok=True)

                kwargs_copy = kwargs.copy()
                kwargs_copy.pop(cache_dir_key, None)
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs_copy)
                tokenizer.save_pretrained(chat_template_cache_dir)

                kwargs_copy = kwargs.copy()
                kwargs_copy.pop(cache_dir_key, None)

                return func(cls, chat_template_cache_dir, kwargs_copy)

        return wrapper

    return decorator


if TYPE_CHECKING:
    from qairt.gen_ai_api.chat.prompt_objects import PromptObject

_logger = get_logger(__name__)


class ChatTemplate(ABC):
    """
    Abstract base class for all chat template implementations.

    This interface ensures all chat templates provide a consistent way to
    apply templates to prompt objects and return formatted prompt strings.
    """

    @abstractmethod
    def apply_template(self, prompt_object: "PromptObject") -> str:
        """
        Apply the chat template to a prompt object and return the final prompt string.

        Args:
            prompt_object: PromptObject containing the user's prompt

        Returns:
            Formatted prompt string ready for model execution

        Raises:
            ValueError: If template application fails
        """
        pass

    @abstractmethod
    def save(self, dest_path: str) -> None:
        """
        Save the chat template to the specified destination path.

        Args:
            dest_path: Destination directory path where the chat template should be saved

        Raises:
            ValueError: If saving fails or template cannot be serialized
        """
        pass


class NullChatTemplate(ChatTemplate, BaseModel):
    """
    Null chat template that passes through the prompt without any formatting.

    This template is used when no chat template is configured, providing
    a simple pass-through behavior for string prompts.
    """

    model_config = ConfigDict(populate_by_name=True)
    type: Literal["null"] = Field(default="null", alias="chat-template-type")

    def apply_template(self, prompt_object: "PromptObject") -> str:
        """
        Apply null template - simply return the prompt as-is.

        Args:
            prompt_object: PromptObject containing the user's prompt

        Returns:
            The original prompt string without any formatting
        """
        return prompt_object.get_prompt()

    def save(self, dest_path: str) -> None:
        """
        Save the null chat template (no-op since there's nothing to save).

        Args:
            dest_path: Destination directory path (unused for null template)
        """
        pass


class HFChatTemplate(ChatTemplate, BaseModel):
    """
    HuggingFace chat template implementation using AutoTokenizer.

    This template loads a HuggingFace tokenizer and uses its built-in
    chat template functionality to format messages.
    """

    model_config = ConfigDict(populate_by_name=True)
    type: Literal["hf"] = Field(default="hf", alias="chat-template-type")
    hf_model_id: Optional[str] = Field(None, alias="hf-model-id")
    path: Optional[str] = None
    tokenizer_kwargs: Optional[Dict] = Field(None, alias="tokenizer-kwargs")
    tokenizer: Optional[Any] = Field(None, exclude=True)

    _validation_cache: Dict[str, bool] = {}

    def __init__(self, tokenizer=None, hf_model_id=None, path=None, tokenizer_kwargs=None, **data):
        """
        Initialize the HFChatTemplate with a tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer with chat template support
            hf_model_id: Original model ID used to create this template (for serialization)
            path: Path where tokenizer was loaded from (for offline support)
            tokenizer_kwargs: Additional tokenizer arguments
        """
        super().__init__(
            hf_model_id=hf_model_id, path=path, tokenizer_kwargs=tokenizer_kwargs, tokenizer=tokenizer, **data
        )

    @model_validator(mode="after")
    def ensure_tokenizer_loaded(self):
        """Ensure tokenizer is loaded when deserializing from saved data."""
        if self.tokenizer is None and self.path and os.path.exists(self.path):
            try:
                kwargs = self.tokenizer_kwargs or {}
                self.tokenizer = AutoTokenizer.from_pretrained(self.path, **kwargs)
            except Exception as e:
                _logger.warning(f"Failed to load tokenizer from {self.path}: {e}")
        return self

    @classmethod
    @cache_hf_tokenizer()
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, kwargs: Optional[Dict] = None
    ) -> "HFChatTemplate":
        """
        Create a HFChatTemplate from a HuggingFace model ID or local tokenizer path.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID (e.g., 'microsoft/Phi-3.5-mini-instruct')
                                         or path to tokenizer directory
            kwargs: Additional arguments forwarded to AutoTokenizer.from_pretrained().

        Returns:
            HFChatTemplate instance

        Raises:
            ValueError: If tokenizer cannot be loaded or doesn't have chat template
        """
        if kwargs is None:
            kwargs = {}

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        path = pretrained_model_name_or_path if os.path.exists(pretrained_model_name_or_path) else None

        instance = cls(
            tokenizer=tokenizer,
            hf_model_id=pretrained_model_name_or_path,
            path=path,
            tokenizer_kwargs=kwargs,
        )
        from qairt.gen_ai_api.chat.prompt_objects import PromptObject

        test_prompt = PromptObject("")
        # Ensure that the loaded tokenizer has a chat template
        instance.apply_template(test_prompt)

        return instance

    def _format_chat_messages(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Format chat messages using the tokenizer's chat template."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded. Cannot format chat messages.")
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def apply_template(self, prompt_object: "PromptObject") -> str:
        """
        Apply HuggingFace chat template to the prompt.

        Args:
            prompt_object: PromptObject containing the user's prompt

        Returns:
            Formatted prompt string using the tokenizer's chat template
        """
        messages = prompt_object.get_prompt(chat_template_config=True)
        return self._format_chat_messages(messages, add_generation_prompt=True)

    def validate_tokenizer_compatibility(
        self, user_tokenizer_path: str | os.PathLike, hf_cache_path: Optional[str | os.PathLike] = None
    ) -> None:
        """
        Compare user tokenizer with downloaded HF chat template tokenizer and warn if different.

        Args:
            user_tokenizer_path: Path to user's tokenizer.json file
            hf_cache_path: Path to cached HF tokenizer directory (None if not cached)
        """
        if not hf_cache_path:
            return

        user_tokenizer_path = str(user_tokenizer_path)
        hf_cache_path = str(hf_cache_path)

        cache_key = f"{user_tokenizer_path}:{hf_cache_path}"
        if cache_key in self._validation_cache:
            return

        if not os.path.exists(user_tokenizer_path):
            self._validation_cache[cache_key] = True
            return

        hf_tokenizer_file = os.path.join(hf_cache_path, "tokenizer.json")
        if not os.path.exists(hf_tokenizer_file):
            self._validation_cache[cache_key] = True
            return

        user_hash = self._get_file_md5(user_tokenizer_path)
        hf_hash = self._get_file_md5(hf_tokenizer_file)

        if user_hash != hf_hash:
            _logger.warning(
                f"Tokenizer mismatch detected:\n"
                f"  This may cause inconsistent behavior between chat formatting and model inference. "
                f"Consider using the same tokenizer for both or verify compatibility."
            )

        self._validation_cache[cache_key] = True

    def save(self, dest_path: str) -> None:
        """
        Save the HuggingFace chat template to the specified destination path.

        Args:
            dest_path: Destination directory path where the chat template should be saved

        Raises:
            ValueError: If saving fails or template cannot be serialized
        """

        if not self.path:
            raise ValueError("Cannot save HFChatTemplate: no source path available")

        if not os.path.exists(self.path):
            raise ValueError(f"Cannot save HFChatTemplate: source path {self.path} does not exist")
        os.makedirs(dest_path, exist_ok=True)

        for item in os.listdir(self.path):
            src_item = os.path.join(self.path, item)
            dst_item = os.path.join(dest_path, item)
            if os.path.isfile(src_item):
                shutil.copy2(src_item, dst_item)

        self.path = str(Path(dest_path).resolve())

    @staticmethod
    def _get_file_md5(file_path: str) -> str:
        """
        Calculate MD5 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash as hexadecimal string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class CustomChatTemplate(ChatTemplate, BaseModel):
    """
    Custom chat template implementation using pure Jinja2.

    This template uses a custom Jinja2 template string and extracts
    special tokens from a base tokenizer for template rendering.
    """

    model_config = ConfigDict(populate_by_name=True)
    type: Literal["custom"] = Field(default="custom", alias="chat-template-type")
    template_string: Optional[str] = Field(None, alias="template-string")
    tokenizer_path: Optional[str] = Field(None, alias="tokenizer-path")
    tokenizer: Optional[Any] = Field(None, exclude=True)

    def __init__(self, tokenizer=None, template_string=None, tokenizer_path=None, **data):
        """
        Initialize the CustomChatTemplate with a tokenizer.

        Args:
            tokenizer: Jinja2 tokenizer with chat template support
            template_string: The Jinja2 template string
            tokenizer_path: Path to the tokenizer
        """
        super().__init__(
            template_string=template_string, tokenizer_path=tokenizer_path, tokenizer=tokenizer, **data
        )

    @model_validator(mode="after")
    def ensure_tokenizer_loaded(self):
        """Ensure tokenizer is loaded when deserializing from saved data."""
        if self.tokenizer is None and self.template_string and self.tokenizer_path:
            try:
                reconstructed = self.from_template_string(self.template_string, self.tokenizer_path)
                self.tokenizer = reconstructed.tokenizer
            except Exception as e:
                _logger.warning(f"Failed to reconstruct tokenizer for CustomChatTemplate: {e}")
        return self

    @classmethod
    def from_template_string(cls, template_string: str, base_tokenizer_path: str) -> "CustomChatTemplate":
        """
        Create a CustomChatTemplate with a custom Jinja2 template string.
        Uses pure Jinja2 processing without HuggingFace dependencies.

        Args:
            template_string: Custom Jinja2 chat template string
            base_tokenizer_path: Path to base tokenizer directory (containing tokenizer.json)

        Returns:
            CustomChatTemplate instance

        Raises:
            ValueError: If tokenizer.json cannot be loaded or template is invalid
        """
        try:
            tokenizer_json_path = None
            if os.path.isfile(base_tokenizer_path) and base_tokenizer_path.endswith(".json"):
                tokenizer_json_path = base_tokenizer_path
            elif os.path.isdir(base_tokenizer_path):
                potential_path = os.path.join(base_tokenizer_path, "tokenizer.json")
                if os.path.exists(potential_path):
                    tokenizer_json_path = potential_path

            if not tokenizer_json_path or not os.path.exists(tokenizer_json_path):
                raise ValueError(f"tokenizer.json not found in {base_tokenizer_path}")

            with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                tokenizer_data = json.load(f)

            special_tokens = cls._extract_special_tokens(tokenizer_data)
            processor = cls._create_jinja2_processor(template_string, special_tokens, base_tokenizer_path)

            return processor

        except Exception as e:
            raise ValueError(f"Failed to create pure Jinja2 processor: {e}")

    @staticmethod
    def _extract_special_tokens(tokenizer_data: dict) -> dict:
        """Extract special tokens from tokenizer.json data."""
        special_tokens = {}

        if "added_tokens" in tokenizer_data:
            for token_info in tokenizer_data["added_tokens"]:
                if isinstance(token_info, dict) and "content" in token_info:
                    content = token_info["content"]
                    special_tokens[content] = content

                    if content in ["<s>", "<|begin_of_text|>", "<|startoftext|>"]:
                        special_tokens["bos_token"] = content
                    elif content in ["</s>", "<|end_of_text|>", "<|endoftext|>"]:
                        special_tokens["eos_token"] = content

        return special_tokens

    @classmethod
    def _create_jinja2_processor(
        cls, template_string: str, special_tokens: dict, base_tokenizer_path: str
    ) -> "CustomChatTemplate":
        """Create a processor that uses pure Jinja2 without HuggingFace tokenizer."""

        class PureJinja2Tokenizer:
            """Minimal tokenizer-like object for pure Jinja2 processing."""

            def __init__(self, template_string: str, special_tokens: dict):
                self.chat_template = template_string
                self.special_tokens = special_tokens

                self.jinja_env = Environment()
                self.jinja_env.globals.update(special_tokens)

                self.template = self.jinja_env.from_string(template_string)

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                """Apply chat template using pure Jinja2."""
                if tokenize:
                    raise ValueError("Tokenization not supported in pure Jinja2 mode")

                try:
                    result = self.template.render(
                        messages=messages, add_generation_prompt=add_generation_prompt, **self.special_tokens
                    )

                    return result
                except Exception as e:
                    raise ValueError(f"Template rendering failed: {e}")

        tokenizer_instance = PureJinja2Tokenizer(template_string, special_tokens)
        instance = cls(
            tokenizer_instance, template_string=template_string, tokenizer_path=base_tokenizer_path
        )
        return instance

    def _format_chat_messages(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Format chat messages using the tokenizer's chat template."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded. Cannot format chat messages.")
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def apply_template(self, prompt_object: "PromptObject") -> str:
        """
        Apply custom Jinja2 template to the prompt.

        Args:
            prompt_object: PromptObject containing the user's prompt

        Returns:
            Formatted prompt string using the custom Jinja2 template
        """
        messages = prompt_object.get_prompt(chat_template_config=True)
        return self._format_chat_messages(messages, add_generation_prompt=True)

    def save(self, dest_path: str) -> None:
        """
        Save the custom chat template to the specified destination path.

        Args:
            dest_path: Destination directory path where the chat template should be saved

        Raises:
            ValueError: If saving fails or template cannot be serialized
        """

        if not self.tokenizer_path:
            raise ValueError("Cannot save CustomChatTemplate: no tokenizer path available")

        if not os.path.exists(self.tokenizer_path):
            raise ValueError(
                f"Cannot save CustomChatTemplate: tokenizer path {self.tokenizer_path} does not exist"
            )
        os.makedirs(dest_path, exist_ok=True)

        if os.path.isfile(self.tokenizer_path) and self.tokenizer_path.endswith(".json"):
            dst_file = os.path.join(dest_path, "tokenizer.json")
            shutil.copy2(self.tokenizer_path, dst_file)
        elif os.path.isdir(self.tokenizer_path):
            for item in os.listdir(self.tokenizer_path):
                src_item = os.path.join(self.tokenizer_path, item)
                dst_item = os.path.join(dest_path, item)
                if os.path.isfile(src_item):
                    shutil.copy2(src_item, dst_item)

        self.tokenizer_path = str(Path(dest_path).resolve())
