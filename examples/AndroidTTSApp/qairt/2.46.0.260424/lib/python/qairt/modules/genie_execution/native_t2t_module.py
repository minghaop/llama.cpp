# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import asyncio
import json
import os
import platform
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from qairt.gen_ai_api.executors.gen_ai_executor import (
    GenerationMetrics,
    TextGenerationResult,
    parse_genie_profile_record,
)
from qairt.modules.genie_execution import genie
from qairt.modules.genie_execution.genie_config import (
    EagletDialog,
    EagletTargetDialogEngine,
    EngineBackendType,
    GenieConfig,
)
from qairt.modules.lora.lora_config import UseCaseRunConfig
from qairt.utils import loggers
from qti.aisw.tools.core.modules.api.backend.utility import HexagonEnvironmentManager


class GenieNativeT2TRunner:
    """
    GenieNativeT2TRunner enables execution in native python environments using wrappers around Genie APIs. Users can utilize
    this API for access to more expressive and fine-grained control over execution in Genie.
    """

    _logger = loggers.get_logger(name=__name__)

    def __init__(self, genie_config: GenieConfig, query_timeout: int = 180):
        """
        The GenieNativeT2TRunner can be constructed from the same GenieConfig used with the command line tool and optionally
        the user may set the query timeout (in seconds).

        Args:
            genie_config (GenieConfig): GenieConfig defining model and execution configuration
            query_timeout (int): Timeout period after which the abort signal will be sent terminating a query
        """
        if platform.system() == "Windows" and "ARMv8" in platform.processor():
            engine = (
                genie_config.dialog.engine
                if not isinstance(genie_config.dialog, EagletDialog)
                else next(
                    engine
                    for engine in genie_config.dialog.engine
                    if isinstance(engine, EagletTargetDialogEngine)
                )
            )
            if engine.backend.type == EngineBackendType.QNN_HTP:
                dsp_arch = "v73"
                try:
                    extensions_path = engine.backend.extensions
                    if isinstance(extensions_path, (str, os.PathLike)):
                        with open(str(engine.backend.extensions), "r") as f:
                            dsp_arch = json.load(f)["devices"][0]["dsp_arch"]
                except Exception:
                    self._logger.warning(
                        f"Failed to parse dsp architecture from extensions config. Falling back to: {dsp_arch}"
                    )
                HexagonEnvironmentManager.activate_hexagon_env(dsp_arch)

        self._profile_config: genie.ProfileConfig = genie.ProfileConfig()
        self._profile: genie.Profile = genie.Profile(self._profile_config)
        self._dialog_config: genie.DialogConfig = genie.DialogConfig(str(genie_config))
        self._dialog_config.bind_profile(self._profile)
        self._dialog: genie.Dialog = genie.Dialog(self._dialog_config)
        self._query_timeout: int = query_timeout
        self._sampler: genie.Sampler = self._dialog.get_sampler()
        self._sampler_callbacks: List[Callable[[np.ndarray], int]] = []
        self._sampler_configs: List[str] = []

    def __del__(self):
        del self._dialog
        del self._dialog_config
        del self._profile
        del self._profile_config
        del self._sampler

    def query(self, prompt: str, *, lora_config: Optional[UseCaseRunConfig] = None) -> TextGenerationResult:
        """
        Executes the provided query

        Args:
            prompt (str): The string prompt to be executed. This could be a pre-formatted text or raw string ready for
                model inference. The native runner does not perform any chat template formatting, JSON parsing,
                or file loading - it passes the prompt directly to the underlying Genie dialog.
            lora_config (UseCaseRunConfig): Configuration used to control how LoRA adapters are applied during model inference.

        Examples:

            .. code-block:: python

                from qairt.modules.genie_execution.genie_config import GenieConfig
                from qairt.modules.genie_execution.native_t2t_module import GenieNativeT2TRunner

                with open("genie_config.json", "r") as f:
                    genie_config = GenieConfig(**json.load(f))

                runner = GenieNativeT2TRunner(genie_config)
                prompt = "<|begin_of_text|>What is the capital of Spain?:"
                result = runner.query(prompt)

        Returns:
            TextGenerationResult: Generated text and execution metrics from native execution
        """
        _ = lora_config  # TODO: AISW-147506 Add support for native execution flow
        out = TextGenerationResult()

        output = []
        started = False

        def capture_output(response: str, code: genie.GenieDialogSentenceCode):
            nonlocal output
            nonlocal started
            started = True
            output.append((response, code))

        with ThreadPoolExecutor(max_workers=1) as executor:
            self._logger.debug(f"querying dialog with: {prompt}")
            future = executor.submit(self._dialog.query, prompt, capture_output)
            try:
                future.result(timeout=self._query_timeout)
            except genie.GenieException as e:
                out.error = str(e)
                return out
            except TimeoutError:
                out.error = f"Query timed out for prompt: {prompt}"
                self._logger.warning(f"Sending abort signal as query timed out for prompt: {prompt}")
                while not started:
                    pass
                self._dialog.signal(genie.GenieDialogAction.ABORT)

        for response, code in output:
            if code in {genie.GenieDialogSentenceCode.BEGIN, genie.GenieDialogSentenceCode.CONTINUE}:
                out.generated_text += response
            if code == genie.GenieDialogSentenceCode.ABORT:
                out.error += genie.GenieDialogSentenceCode.ABORT.name

        out.metrics = self.extract_profiling_data()

        return out

    async def stream_query(self, prompt: str, q: asyncio.Queue) -> TextGenerationResult:
        """
        Executes a prompt in streaming mode, sending output chunks to a queue as they are generated.

        This function runs the blocking LLM query in a background thread using run_in_executor, allowing the main
        event loop to remain responsive. As the model generates output, it invokes capture_output to push
        each chunk into the provided asyncio queue. A `None` value is sent to the queue to signal
        the end of the stream.

        Args:
            prompt (str): The input prompt to send to the LLM.
            q (asyncio.Queue): An asyncio queue used to stream output chunks back to the caller.

        Returns:
            TextGenerationResult: An object containing the full generated text and any execution metrics or errors.

        Raises:
            asyncio.TimeoutError: If the query exceeds the timeout duration.
        """

        loop = asyncio.get_running_loop()
        out = TextGenerationResult()
        generated_text = []

        def capture_output(response: str, code: genie.GenieDialogSentenceCode):
            nonlocal generated_text
            if code.name in {"BEGIN", "CONTINUE"}:
                generated_text.append(response)
                loop.call_soon_threadsafe(q.put_nowait, response)
            elif code.name == "END":
                loop.call_soon_threadsafe(q.put_nowait, None)

        def run_query():
            try:
                self._dialog.query(prompt, capture_output)
            except Exception as e:
                self._logger.warning(f"Encountered error {e} while attempting to stream query.")
                self._dialog.signal(genie.GenieDialogAction.ABORT)
                out.error = genie.GenieDialogSentenceCode.ABORT.name
                loop.call_soon_threadsafe(q.put_nowait, None)

        self._logger.debug(f"Streaming query with: {prompt}")
        try:
            await asyncio.wait_for(loop.run_in_executor(None, run_query), timeout=self._query_timeout)
        except asyncio.TimeoutError:
            self._logger.warning(
                f"Sending abort signal as query timed out ({self._query_timeout} seconds) for prompt: {prompt}"
            )
            self._dialog.signal(genie.GenieDialogAction.ABORT)
            out.error = f"Query timed out for prompt: {prompt}"
            loop.call_soon_threadsafe(q.put_nowait, None)

        out.generated_text = "".join(generated_text)
        out.metrics = self.extract_profiling_data()
        return out

    def save_dialog(self, save_dir: Union[str, os.PathLike]) -> None:
        """
        Stores the current state of the genie dialog

        Args:
            save_dir (Union[str, os.PathLike]): Location to save the dialog
        """
        path = Path(save_dir)
        if path.is_file():
            raise FileExistsError(
                f"Provided save location is an existing file {path}. Please provide a directory"
            )
        path.mkdir(exist_ok=True, parents=True)

        self._dialog.save(str(path))
        self._logger.info(f"Saved dialog to {path}")

    def restore_dialog(self, saved_dialog: Union[str, os.PathLike]) -> "GenieNativeT2TRunner":
        """
        Restores a saved genie dialog state

        Args:
            saved_dialog (Union[str, os.PathLike]): Path to saved dialog state to restore

        Returns:
            GenieNativeT2TRunner: Returns self after restoration
        """
        path = Path(saved_dialog)
        if not path.is_dir():
            raise NotADirectoryError(f"Provided path to saved dialog {path} is not an existing directory.")

        self.reset_dialog()
        self._dialog.restore(str(path))
        self._logger.info(f"Restored dialog from {path}")
        return self

    def reset_dialog(self) -> "GenieNativeT2TRunner":
        """
        Resets dialog state to remove context accumulated from queries

        Returns:
            self: Returns self after resetting dialog state
        """
        self._dialog.reset()
        self._logger.info(f"Reset dialog")
        return self

    def register_sampler_callback(
        self, name: str, callback: Callable[[np.ndarray], int]
    ) -> "GenieNativeT2TRunner":
        """
        Register sampler callback function

        Args:
            name (str): Name of the callback. Passed in a sampler config to set the desired sampler callback
            callback (Callable[[np.ndarray], int]): Sampler callback to select the next token given logits
        Returns:
            self: Returns self after registering callback
        """
        self._sampler.register_callback(name, callback)
        self._sampler_callbacks.append(callback)
        self._logger.info("Registered sampler callback")
        return self

    def apply_sampler_config(self, config: str) -> "GenieNativeT2TRunner":
        """
        Apply a sampler config either setting sampler parameters for the default sampler or supplying the
        name of a registered sampler callback function

        Args:
            config (str): json string representing the sampler config to apply
        Returns:
            self: Returns self after applying the sampler config
        """
        sampler_config = genie.SamplerConfig(config)
        self._sampler.apply_config(sampler_config)
        self._sampler_configs.append(sampler_config)
        self._logger.info("Applied sampler config")
        return self

    def extract_profiling_data(self) -> Optional[GenerationMetrics]:
        try:
            profile_record = json.loads(self._profile.get_json_data())
            return parse_genie_profile_record(profile_record)

        except Exception as e:
            self._logger.warning(f"Failed to parse the profiling record:\n{str(e)}")
            return None
