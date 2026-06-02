# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

from qairt.api.configs.common import AISWBaseModel
from qairt.gen_ai_api.chat.prompt_objects import MessagesObject, PromptObject
from qairt.gen_ai_api.configs.gen_ai_config import GenAIConfig


class GenerationMetrics(AISWBaseModel):
    init_time: Optional[int] = None
    """Time to load the model, before prompt processing begins"""
    prompt_processing_time: Optional[int] = None
    """Microseconds between init and token generation, while processing the prompt."""
    prompt_processing_rate: Optional[float] = None
    """tokens per second.  Tokens in the prompt divided by prompt processing time"""
    token_generation_time: Optional[int] = None
    """Microseconds prompt processing and final response"""
    token_generation_rate: Optional[float] = None
    """tokens per second.  Tokens in the response divided by token generation time"""
    time_to_first_token: Optional[int] = None
    """Microseconds from start of prompt processing to first token generated"""
    token_acceptance_rate: Optional[float] = None
    """average tokens accepted per inference step"""
    adapter_switch_time: Optional[int] = None
    """Microseconds to switch between LoRA adapaters."""

    def get_metrics_dict(self) -> Dict[str, str]:
        """
        Returns an ordered dictionary of metric names to formatted values.
        Only includes metrics that are set (non-None and non-zero for optional metrics).
        The order of insertion is preserved.

        Returns:
            OrderedDict mapping metric names to their formatted string values.
        """
        metrics = OrderedDict()

        # Timing metrics (in desired display order)
        metrics["Init Time"] = f"{self.init_time or 0.0} us"
        metrics["Prompt Processing Time"] = f"{self.prompt_processing_time or 0.0} us"
        metrics["Time to First Token"] = f"{self.time_to_first_token or 0.0} us"
        metrics["Token Generation Time"] = f"{self.token_generation_time or 0.0} us"

        # Optional timing metric
        if self.adapter_switch_time and self.adapter_switch_time != 0.0:
            metrics["Adapter Switch Time"] = f"{self.adapter_switch_time} us"

        # Rate metrics
        metrics["Prompt Processing Rate"] = f"{self.prompt_processing_rate or 0} toks/sec"
        metrics["Token Generation Rate"] = f"{self.token_generation_rate or 0} toks/sec"

        # Optional per-inference metric
        if self.token_acceptance_rate and self.token_acceptance_rate != 0.0:
            metrics["Token Acceptance Rate"] = f"{self.token_acceptance_rate} toks/inference"

        return metrics

    def __str__(self):
        metrics_dict = self.get_metrics_dict()

        # Build metrics list in the order from get_metrics_dict()
        lines = []

        for key, value in metrics_dict.items():
            lines.append(f"  {key} = {value} \n")

        # Calculate the maximum line length
        if lines:
            max_length = max(len(line.rstrip()) for line in lines)
        else:
            max_length = 40  # Default minimum width

        # Create header that spans the full width
        header = f"{'-' * max_length}\n"
        title = "Metrics"
        title_line = f"{title.center(max_length)}\n"

        return header + title_line + header + "\n" + "".join(lines)

    def print(self, console: Optional[Any] = None) -> None:
        """
        Prints the metrics as a simple table.

        Args:
            console: A rich Console instance to print to. Defaults to None, which creates
                a Console writing to stdout.
        """
        # import here to avoid hard dependency on rich
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            # print warning and fallback to simple print
            print("Warning: 'rich' library is not installed. Falling back to simple print.")
            print(self)
            return

        if console is None:
            console = Console()
        table = Table(show_header=True)

        table.add_column("Metric")
        table.add_column("Value", justify="right")

        # Get metrics and add rows in order
        metrics_dict = self.get_metrics_dict()
        for metric_name, metric_value in metrics_dict.items():
            table.add_row(metric_name, metric_value)

        console.print(table)


def _extract_metric_value(event: Dict[str, Any], metric_name: str) -> Optional[Union[int, float]]:
    """Helper to safely extract metric value from an event."""
    if metric_name in event and "value" in event[metric_name]:
        return event[metric_name]["value"]
    return None


def _parse_execution_metrics(execution_event: Dict[str, Any], metrics: GenerationMetrics) -> None:
    """Parse common execution metrics from either pipeline or dialog format."""
    prompt_rate = _extract_metric_value(execution_event, "prompt-processing-rate")
    if prompt_rate is not None:
        metrics.prompt_processing_rate = prompt_rate
        num_prompt_tokens = _extract_metric_value(execution_event, "num-prompt-tokens")
        if num_prompt_tokens is not None:
            # Calculate prompt processing time: tokens / (tokens/sec) * 1,000,000 us/sec
            metrics.prompt_processing_time = int(1000000 * num_prompt_tokens / prompt_rate)

    token_gen_time = _extract_metric_value(execution_event, "token-generation-time")
    if token_gen_time is not None:
        metrics.token_generation_time = int(token_gen_time)

    token_gen_rate = _extract_metric_value(execution_event, "token-generation-rate")
    if token_gen_rate is not None:
        metrics.token_generation_rate = token_gen_rate

    ttft = _extract_metric_value(execution_event, "time-to-first-token")
    if ttft is not None:
        metrics.time_to_first_token = int(ttft)

    # Token acceptance rate (typically only in dialog format)
    token_accept_rate = _extract_metric_value(execution_event, "token-acceptance-rate")
    if token_accept_rate is not None:
        metrics.token_acceptance_rate = token_accept_rate


class _FormatMapping(TypedDict):
    """Type definition for profile format mapping entries."""

    component_type: str
    create_event: str
    execute_event: str
    lora_event: str
    init_metric: str
    init_is_direct: bool


_PIPELINE_DIALOG_PROFILE_FORMAT_MAPPINGS: Dict[str, _FormatMapping] = {
    "pipeline": {
        "component_type": "pipeline",
        "create_event": "GeniePipeline_create",
        "execute_event": "GeniePipeline_execute",
        "lora_event": "GeniePipeline_applyLora",
        "init_metric": "duration",  # init time stored directly as duration
        "init_is_direct": True,  # duration is a direct field, not nested in an object
    },
    "dialog": {
        "component_type": "dialog",
        "create_event": "GenieDialog_create",
        "execute_event": "GenieDialog_query",
        "lora_event": "GenieDialog_applyLora",
        "init_metric": "init-time",  # init time stored in an object
        "init_is_direct": False,  # init time is nested in an object
    },
}


def parse_genie_profile_record(profile_record: Dict[str, Any]) -> GenerationMetrics:
    metrics = GenerationMetrics()
    if "components" not in profile_record:
        return metrics

    for format_name, mapping in _PIPELINE_DIALOG_PROFILE_FORMAT_MAPPINGS.items():
        components = [x for x in profile_record["components"] if x["type"] == mapping["component_type"]]
        if not components:
            continue

        events = components[0]["events"]

        create_events = [x for x in events if x["type"] == mapping["create_event"]]
        if create_events:
            if mapping["init_is_direct"]:
                if mapping["init_metric"] in create_events[0]:
                    metrics.init_time = int(create_events[0][mapping["init_metric"]])
            else:
                init_time = _extract_metric_value(create_events[0], mapping["init_metric"])
                if init_time is not None:
                    metrics.init_time = int(init_time)

        execute_events = [x for x in events if x["type"] == mapping["execute_event"]]
        if execute_events:
            _parse_execution_metrics(execute_events[-1], metrics)

        lora_events = [x for x in events if x["type"] == mapping["lora_event"]]
        if lora_events:
            adapter_switch = _extract_metric_value(lora_events[0], "lora-adapter-switching-time")
            if adapter_switch is not None:
                metrics.adapter_switch_time = int(adapter_switch)

        # return when a valid matching format is found
        return metrics

    return metrics


def process_prompt(prompt: Union[str, List[Dict[str, str]], Path], genai_config: GenAIConfig) -> str:
    """
    Process prompt input and return formatted prompt string.

    Args:
        prompt: Prompt input. Can be a raw string, file path, or list of dicts for chat format.
        genai_config: GenAI configuration containing chat template settings.

    Returns:
        Formatted prompt string ready for execution
    """
    prompt_obj: Union[PromptObject, MessagesObject]

    if prompt is None:
        prompt_obj = PromptObject(None)
    elif isinstance(prompt, list):
        prompt_obj = MessagesObject(prompt)
    elif isinstance(prompt, Path):
        prompt_obj = MessagesObject.fromJson(str(prompt))
    else:
        try:
            prompt_obj = MessagesObject.fromJson(prompt)
        except (json.JSONDecodeError, ValueError, IOError):
            prompt_obj = PromptObject(prompt)

    chat_template = genai_config.chat_template
    if chat_template:
        return chat_template.apply_template(prompt_obj)
    else:
        return prompt_obj.get_prompt()


class GenerationExecutionResult(AISWBaseModel):
    """Base class for generation execution results."""

    output: str = ""
    """Raw output from generation"""
    error: str = ""
    """Raw error response from generation (empty on success)"""
    metrics: Optional[GenerationMetrics] = None
    """parsed metrics from the response."""


class TextGenerationResult(GenerationExecutionResult):
    """Result class specifically for text-to-text generation."""

    generated_text: str = ""
    """parsed response - the generated response (minus metrics)"""

    def print(self, metrics: bool = True, console: Optional[Any] = None) -> None:
        """
        Prints the generated text rendered as Markdown, followed optionally by metrics.

        Args:
            metrics: If True, prints generation metrics after the text. Defaults to True.
            console: A rich Console instance to print to. Defaults to None, which creates
                a Console writing to stdout. To route output to a file, pass
                ``Console(file=open("output.txt", "w"))``.
        """
        try:
            from rich.console import Console as RichConsole
            from rich.markdown import Markdown
            from rich.panel import Panel
        except ImportError:
            print("Warning: 'rich' library is not installed. Falling back to simple print.")
            print(self.generated_text)
            if metrics and self.metrics:
                self.metrics.print()
            return

        if console is None:
            console = RichConsole()

        console.print(Panel(Markdown(self.generated_text), title="Generated Text", expand=False))

        if metrics:
            if self.metrics:
                self.metrics.print(console=console)
            else:
                console.print("Metrics were not generated")


class GenAIExecutor(ABC):
    @abstractmethod
    def prepare_environment(self) -> "GenAIExecutor":
        """
        Prepares artifacts for execution on target
        """
        pass

    @abstractmethod
    def clean_environment(self) -> "GenAIExecutor":
        """
        Removes artifacts from target environment
        """
        pass
