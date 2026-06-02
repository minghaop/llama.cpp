# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union


class PromptObject:
    """Handles string prompts with default fallback."""

    DEFAULT_SYSTEM_MESSAGE = "Be helpful but try to limit answers to 40 words."
    DEFAULT_USER_MESSAGE = "What can I do with a glass jar?"

    def __init__(self, prompt: Optional[str] = None):
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"Prompt must be a string or None, got {type(prompt)}")
        self.prompt = prompt or self.DEFAULT_USER_MESSAGE

    def get_prompt(self, chat_template_config=None):
        """Returns the final prompt ready for execution."""
        if chat_template_config:
            return [
                {"role": "system", "content": self.DEFAULT_SYSTEM_MESSAGE},
                {"role": "user", "content": self.prompt},
            ]

        # For default prompts, use legacy format
        if self.prompt == self.DEFAULT_USER_MESSAGE:
            return (
                "<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>{self.DEFAULT_SYSTEM_MESSAGE}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>{self.prompt}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )

        return self.prompt


class MessagesObject(PromptObject):
    """Handles OpenAI message format prompts."""

    def __init__(self, messages: List[Dict[str, str]]):
        # Validate messages format
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Messages must be a non-empty list")

        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            if "role" not in message or "content" not in message:
                raise ValueError(f"Message {i} must have 'role' and 'content' keys")

        self.messages = messages

    @classmethod
    def fromJson(cls, messages_input: Union[str, Path]) -> "MessagesObject":
        """
        Create a MessagesObject from JSON string or file path.

        Args:
            messages_input: JSON string or path to JSON file containing messages

        Returns:
            MessagesObject: Instance created from the parsed JSON

        Raises:
            json.JSONDecodeError: If JSON parsing fails
            IOError: If file reading fails
            ValueError: If messages format is invalid
        """
        if isinstance(messages_input, Path) or os.path.isfile(messages_input):
            with open(messages_input, "r") as f:
                messages = json.load(f)
        else:
            messages = json.loads(messages_input)

        return cls(messages)

    def get_prompt(self, chat_template_config=None):
        """Returns messages as-is - requires chat template for processing."""
        if chat_template_config:
            return self.messages

        # Messages format requires a chat template - no fallback
        raise ValueError("Messages format input requires a chat template to be configured.")
