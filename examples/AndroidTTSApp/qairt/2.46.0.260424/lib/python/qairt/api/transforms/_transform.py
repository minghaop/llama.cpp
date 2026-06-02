# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


import os

import onnx
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from qairt.api.configs.common import BackendType
from qairt.api.transforms.model_transformer_config import (
    ModelTransformerConfig,
    QuantizationStage,
    SplitModelConfig,
)
from qairt.optimizer.onnx import adapt_moe
from qairt.utils.loggers import get_logger
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_model import OnnxModel


def transform(
    model: str | os.PathLike | onnx.ModelProto,
    backend: BackendType = BackendType.HTP,
    quantization_stage: QuantizationStage | None = None,
    encodings: str | os.PathLike | None = None,
    lora_adapters_path: str | os.PathLike | None = None,
    lora_tensor_names_path: str | os.PathLike | None = None,
    **transforms,
) -> list[OnnxModel]:
    """
    Transforms an ONNX model by performing applicable transforms that Gen AI builder expects
      MHA2SHA (MultiHead Attention to Single Head Attention) and/or model splitting.

    Args:
        model (str | PathLike | onnx.ModelProto): The ONNX model to be transformed, either as a
            file path or a ModelProto object.
        backend (Optional[BackendType]): The backend type for which the model is being transformed.
            Defaults to BackendType.HTP.
        quantization_stage (Optional[QuantizationStage]): The quantization stage for the
            transformation. Defaults to QuantizationStage.POST_QUANT. If post_quant is passed,
            then MHA2SHA and model splitting will be performed.
        encodings (Optional[str | PathLike]): Encodings to be used for the
            transformation. Can be a str or a file path. Defaults to None.
        lora_adapters_path: path to a yaml file declaring the LoRA adapters (use cases).
        lora_tensor_names_path: path to a text file containing the LoRA tensor names.

        transforms: Transform configurations. Defaults to split_model and mha_to_sha if no arguments are passed.
            split_model: SplitModelConfig or serialized (as dictionary),
                See :class:`qairt.api.transforms.model_transformer_config` for more details.
            mha_config: MhaConfig or serialized (as dictionary).
                See :class:`qairt.api.transforms.model_transformer_config` for more details.
            adapt_moe: See :function:`qairt.optimizer.onnx.adapt_moe` for valid options.
                Pass a dict of kwargs to forward to ``adapt_moe``, or an empty dict / ``True``
                to run with defaults. If omitted, MoE transformation is skipped.

    Examples:
        .. code-block:: python

        import qairt
        fw_model = "path/to/model"
        transformed_model = transform(fw_model,
                                      backend=BackendType.HTP,
                                      quantization_stage=QuantizationStage.POST_QUANT)

    Returns:
        list[OnnxModel]: An list of OnnxModel objects containing the split/transformed model and encodings.
    """

    _transform_logger = get_logger("qairt.transform")

    # TODO: Deprecate Model transformer config in favor of API signature mappings
    # Parse relevant kwargs from transforms using ModelTransformerConfig
    # Don't add new dataclasses to model transformer config

    config = ModelTransformerConfig.from_dict(transforms)

    if isinstance(model, onnx.ModelProto) and isinstance(encodings, dict):
        onnx_model = OnnxModel(
            model=model,
            encodings=encodings,
            lora_adapters=lora_adapters_path,
            lora_tensor_names=lora_tensor_names_path,
        )
    else:
        onnx_model = OnnxModel.load(
            model_path=model,
            encodings_path=encodings,
            lora_adapters_path=lora_adapters_path,
            lora_tensor_names_path=lora_tensor_names_path,
        )

    if quantization_stage is None:
        _transform_logger.warning("`quantization_stage` not set. Defaulting to POST_QUANT")
        quantization_stage = QuantizationStage.POST_QUANT

    # Apply transformations based on backend and quantization stage
    match backend:
        case BackendType.HTP:
            match quantization_stage:
                case QuantizationStage.POST_QUANT:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn("{task.completed}/{task.total}"),
                        TimeElapsedColumn(),
                    ) as _progress:
                        split_task = _progress.add_task("[bold cyan]Splitting model...", total=None)
                        splits = onnx_model.split(
                            num_splits=config.split_model.num_splits,
                            split_embedding=config.split_model.split_embedding,
                            split_lm_head=config.split_model.split_lm_head,
                        )
                        _progress.update(
                            split_task,
                            description=f"[green]Split complete:[/green] {len(splits)} parts",
                            total=1,
                            completed=1,
                        )
                        _progress.remove_task(split_task)

                        task = _progress.add_task("Transforming splits...", total=len(splits))
                        for i, _split in enumerate(splits):
                            _progress.update(task, description=f"Transforming split {i + 1}/{len(splits)}...")
                            if config.adapt_moe is not None:
                                kwargs = config.adapt_moe if isinstance(config.adapt_moe, dict) else {}
                                adapt_moe(_split.graph_context, **kwargs)
                            if config.mha_config:
                                _split.mha2sha_v2(**config.mha_config.__dict__)
                            else:
                                _split.mha2sha_v2()
                            _progress.advance(task)
                        _progress.remove_task(task)
                    return splits
                case QuantizationStage.PRE_QUANT:
                    raise NotImplementedError(
                        "Pre-quantization transformations are not currently supported through this API."
                    )
                case _:
                    raise ValueError(
                        f"Invalid value for quantization_stage: {quantization_stage}. Expected one of: QuantizationStage.PRE_QUANT or QuantizationStage.POST_QUANT"
                    )
        case _:
            raise NotImplementedError(f"Backend type {backend} not supported")
