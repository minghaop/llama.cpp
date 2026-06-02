# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
from pathlib import Path

from qti.aisw.accuracy_debugger.common_config import EncodingInputConfig
from qti.aisw.accuracy_debugger.compare_encodings.compare_encodings_utils import (
    VALID_BITWIDTHS,
    ComparisonStatus,
    FieldStatus,
    compare_dtype,
    compare_scale_offset,
    get_comparison_structure,
    initialize_comparison_dict,
    updated_comparison_status,
)
from qti.aisw.accuracy_debugger.encodings.encodings import Encoding, ModelEncoding, TensorEncoding
from qti.aisw.accuracy_debugger.encodings.encodings_utils import (
    EncodingType,
    TensorType,
    get_encodings_version,
)
from qti.aisw.accuracy_debugger.encodings_converter.qairt_encodings_converter import (
    QairtEncodingsConverter,
)
from qti.aisw.accuracy_debugger.utils.file_utils import dump_csv, dump_json, read_json
from qti.aisw.accuracy_debugger.utils.graph_utils import (
    get_common_parent_activations,
    get_subgraph,
    get_supergroup_activations,
)
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class CompareEncodings:
    """Implements CompareEncodings class"""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initializes the Compare Encodings class

        Args:
            logger: Python Logger object
        """
        self._logger = logger or QAIRTLogger.register_area_logger(
            area=LogAreas.register_log_area("Compare Encodings"),
            level="INFO",
            formatter_val="simple",
            handler_list=["dev_console"],
        )

    def _run_encodings_converter(
        self, encoding_config: EncodingInputConfig, framework_model_path: Path, output_dir: Path
    ) -> tuple[QairtEncodingsConverter, Encoding]:
        """Creates, executes encodings converter and dumps the converted encodings

        Args:
            encoding_config (EncodingInputConfig): encoding config for qairt encodings
            framework_model_path (Path): path to the framework model
            output_dir (Path): path to the output directory to dump converted encodings

        Returns:
            (QairtEncodingsConverter, Encoding): Following things will be returned:
                1. encoding converter object
                2. Encoding object for the converted encodings
        """
        encoding_converter = QairtEncodingsConverter(
            framework_model_path=framework_model_path,
            quantized_dlc_path=str(encoding_config.quantized_dlc_path),
            working_dir=output_dir,
            logger=self._logger,
        )
        model_encoding = encoding_converter.create_subgraph_encodings()
        converted_encoding_path = output_dir.joinpath(encoding_config.encoding_file_path.name)
        model_encoding.dump(
            version=get_encodings_version(read_json(encoding_config.encoding_file_path)),
            file_path=converted_encoding_path,
        )

        return encoding_converter, model_encoding

    def run(
        self,
        encoding_config1: EncodingInputConfig,
        encoding_config2: EncodingInputConfig,
        output_dir: Path,
        framework_model_path: Path | None = None,
        scale_threshold: float = 1e-3,
    ) -> None:
        """Compares primary_encoding against reference encoding and vice-versa

        Args:
            encoding_config1 (EncodingInputConfig): Aimet or Qairt encoding details
            encoding_config2 (EncodingInputConfig): Aimet or Qairt encoding details
            output_dir (Path): path to the output directory to dump the analysis results
            framework_model_path (Path | None): path to the framework model. If passed
                along side with quantized dlc for any of the encoding_config, it performs following
                operations on the qairt encodings file:
                1.  Propagates convert_ops encodings to the its parent op considering the fact that
                    parent op exists in the framework model
                2.  Resolves any activation name changes done. For e.g. matmul+add in framework
                    model becomes fc in the dlc graph and the tensor name gets _fc suffix.
                It also performs supergroup mapping which maps each QNN tensor to a set of tensors
                in framework graph which got fused to form the fused QNN op.
            scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3
                For e.g:
                scale1=0.5, scale2=0.01. We compare scale1 and scale2 as:
                abs(scale1-scale2)<(min(scale1, scale2)*scale_threshold). This ensures that bound is
                maintained by the lowest scale value among the given two scales.

        Raises:
            FileNotFoundError: If framework model does not exists at framework_model_path
            ValueError: If scale_threshold <=0
        """
        # Check if the framework_model_path exists
        if framework_model_path:
            framework_model_path = Path(framework_model_path)
            if not framework_model_path.exists():
                raise FileNotFoundError("framework_model_path is invalid")

        # Check scale_threshold value is >0
        if scale_threshold <= 0:
            raise ValueError("scale_threshold must be > 0")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        encoding_converter1, encoding_converter2 = None, None
        model_encoding1 = ModelEncoding()
        model_encoding2 = ModelEncoding()

        # Model encoding can come either via dlc or json file
        model_encoding1.load(artifact=encoding_config1.encoding_path)
        model_encoding2.load(artifact=encoding_config2.encoding_path)

        # Run encoding_converter if encoding_config1.quantized_dlc_path and framework is passed.
        if encoding_config1.quantized_dlc_path and framework_model_path:
            encoding_name1 = model_encoding1.name
            encoding_dir1 = output_dir.joinpath(model_encoding1.name)
            encoding_converter1, model_encoding1 = self._run_encodings_converter(
                encoding_config=encoding_config1,
                framework_model_path=framework_model_path,
                output_dir=encoding_dir1,
            )
            model_encoding1.name = encoding_name1

        # Run encoding_converter if encoding_config2.quantized_dlc_path and framework is passed.
        if encoding_config2.quantized_dlc_path and framework_model_path:
            encoding_name2 = model_encoding2.name
            encoding_dir2 = output_dir.joinpath(model_encoding2.name)
            encoding_converter2, model_encoding2 = self._run_encodings_converter(
                encoding_config=encoding_config2,
                framework_model_path=framework_model_path,
                output_dir=encoding_dir2,
            )
            model_encoding2.name = encoding_name2

        # Compare activation and param encodings
        self._compare(
            model_encoding1=model_encoding1,
            model_encoding2=model_encoding2,
            output_dir=output_dir,
            scale_threshold=scale_threshold,
        )

        # Map an activation tensor in encoding2 to a supergroup in encoding1 if encoding_converter1
        # is not None
        if encoding_converter1:
            self._map_supergroup(
                encodings_converter=encoding_converter1,
                model_encoding=model_encoding2,
                output_dir=output_dir,
            )

        # Map an activation tensor in encoding1 to a supergroup in encoding2 if encoding_converter2
        # is not None
        if encoding_converter2:
            self._map_supergroup(
                encodings_converter=encoding_converter2,
                model_encoding=model_encoding1,
                output_dir=output_dir,
            )

    def _compare_common_tensor_encodings(
        self, tensor_encoding1: TensorEncoding, tensor_encoding2: TensorEncoding
    ) -> tuple[dict, str]:
        """Compares two tensor encodings
        Args:
            tensor_encoding1 (TensorEncoding): TensorEncoding object for tensor1
            tensor_encoding2 (TensorEncoding): TensorEncoding object for tensor2

        Returns:
            (dict, ComparisonStatus): tuple of:
                1. dictionary of comparison info with keys: dtype, is_symm, channels, scale, offset
                2. ComparisonStatus Enum of the encodings comparision
        """
        info = initialize_comparison_dict()
        comparison_status = ComparisonStatus.SUCCESS

        # ------------------------------------------------------------------------------------------
        # Compare the dtype
        # ------------------------------------------------------------------------------------------
        compare_info, comparison_status = compare_dtype(
            dtype1=tensor_encoding1.dtype, dtype2=tensor_encoding2.dtype
        )
        info["dtype"] = compare_info["dtype"]

        # ------------------------------------------------------------------------------------------
        # Compare the is_symmetric
        # ------------------------------------------------------------------------------------------
        # Compare only if is_sym field are present. Incase of float encodings, it may not be present
        if tensor_encoding1.is_symm and tensor_encoding2.is_symm:
            if tensor_encoding1.is_symm != tensor_encoding2.is_symm:
                info["is_symm"] = FieldStatus.IS_SYMMETRIC_NOT_SAME.value.format(
                    tensor_encoding1.is_symm, tensor_encoding2.is_symm
                )
                comparison_status = updated_comparison_status(
                    comparison_status, ComparisonStatus.WARNING
                )
            else:
                info["is_symm"] = FieldStatus.SAME.value

        # ------------------------------------------------------------------------------------------
        # Compare the bitwidth
        # ------------------------------------------------------------------------------------------
        # If bitwidths are not same, it may not conclude that two encodings being different as
        # encoding with bitwidth b1 can be algebrically converted to encoding with bitwidth b2.
        if (
            tensor_encoding1.bitwidth in VALID_BITWIDTHS
            and tensor_encoding2.bitwidth in VALID_BITWIDTHS
        ):
            if tensor_encoding1.bitwidth != tensor_encoding2.bitwidth:
                info["bitwidth"] = FieldStatus.BITWIDTH_NOT_SIMILAR.value.format(
                    tensor_encoding1.bitwidth, tensor_encoding2.bitwidth
                )
                comparison_status = updated_comparison_status(
                    comparison_status, ComparisonStatus.WARNING
                )
            else:
                info["bitwidth"] = FieldStatus.SAME.value
        else:
            comparison_status = ComparisonStatus.ERROR
            info["bitwidth"] = FieldStatus.INVALID_BITWIDTH.value

        return info, comparison_status

    def _compare_per_tensor_or_channel_tensor_encodings(
        self,
        tensor_encoding1: TensorEncoding,
        tensor_encoding2: TensorEncoding,
        scale_threshold: float = 1e-3,
    ) -> tuple[dict, str]:
        """Compares two tensor encodings
        Args:
            tensor_encoding1 (TensorEncoding): TensorEncoding object for tensor1
            tensor_encoding2 (TensorEncoding): TensorEncoding object for tensor2
            scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

        Returns:
            (dict, str): tuple of:
                1. dictionary of comparison info with keys: dtype, is_symm, channels, scale, offset
                2. ComparisonStatus value of the encodings comparision
        """
        info, comparison_status = self._compare_common_tensor_encodings(
            tensor_encoding1=tensor_encoding1, tensor_encoding2=tensor_encoding2
        )

        # ------------------------------------------------------------------------------------------
        # Compare the channels, scale, and offset
        # ------------------------------------------------------------------------------------------

        # The following scenarios can happen:
        # 1. number of channels are not same => both encodings are not same
        # 2. If bitwidth of both encodings are in valid supported bitwidths.
        # 2.a. If Scale and offsets are present in both encodings then check whether channels, scale
        #      and offsets are same or not for both encodings
        # 2.b. Scale and offset not present in one of the encodings
        if (
            tensor_encoding1.bitwidth in VALID_BITWIDTHS
            and tensor_encoding2.bitwidth in VALID_BITWIDTHS
        ):
            if (
                tensor_encoding1.scale
                and tensor_encoding1.offset
                and tensor_encoding2.scale
                and tensor_encoding2.offset
            ):
                compare_info, _comparison_status = compare_scale_offset(
                    tensor_encoding1=tensor_encoding1,
                    tensor_encoding2=tensor_encoding2,
                    scale_threshold=scale_threshold,
                )
                info["channels"] = compare_info["channels"]
                info["scale"] = compare_info["scale"]
                info["offset"] = compare_info["offset"]
                comparison_status = updated_comparison_status(comparison_status, _comparison_status)
            elif (tensor_encoding1.offset and tensor_encoding1.scale) != (
                tensor_encoding2.offset and tensor_encoding2.scale
            ):
                info["offset"] = FieldStatus.OFFSET_NOT_PRESENT.value
                info["scale"] = FieldStatus.SCALE_NOT_PRESENT.value
                comparison_status = ComparisonStatus.ERROR

        return info, comparison_status.value

    def _compare_lpbq_tensor_encodings(
        self,
        tensor_encoding1: TensorEncoding,
        tensor_encoding2: TensorEncoding,
        scale_threshold: float = 1e-3,
    ) -> tuple[dict, str]:
        """Compares two tensor encodings
        Args:
            tensor_encoding1 (TensorEncoding): TensorEncoding object for tensor1
            tensor_encoding2 (TensorEncoding): TensorEncoding object for tensor2
            scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

        Returns:
            (dict, str): tuple of:
                1. dictionary of comparison info with keys: dtype, is_symm, channels, scale, offset
                2. ComparisonStatus value of the encodings comparision
        """
        info, comparison_status = self._compare_common_tensor_encodings(
            tensor_encoding1=tensor_encoding1, tensor_encoding2=tensor_encoding2
        )

        # ------------------------------------------------------------------------------------------
        # Compare per_block_float_scale
        # ------------------------------------------------------------------------------------------
        _info = FieldStatus.SAME.value
        _comparison_status = ComparisonStatus.SUCCESS
        if tensor_encoding1.channels == tensor_encoding2.channels:
            info["channels"] = FieldStatus.SAME.value
            for index, scale in enumerate(zip(tensor_encoding1.scale, tensor_encoding2.scale)):
                s1, s2 = scale
                threshold = scale_threshold * min(s1, s2)
                if abs(s1 - s2) > threshold:
                    _info = FieldStatus.SCALE_NOT_SAME.value.format(threshold, s1, s2, index)
                    _comparison_status = ComparisonStatus.ERROR
                    break
        else:
            _info = FieldStatus.NOT_COMPARED.value
            info["channels"] = FieldStatus.NUM_CHANNELS_NOT_SAME.value.format(
                tensor_encoding1.channels, tensor_encoding2.channels
            )
            _comparison_status = ComparisonStatus.ERROR

        info["scale"] = _info
        comparison_status = updated_comparison_status(comparison_status, _comparison_status)

        # ------------------------------------------------------------------------------------------
        # Compare per_block_int_scale
        # ------------------------------------------------------------------------------------------
        _info = FieldStatus.SAME.value
        _comparison_status = ComparisonStatus.SUCCESS
        if len(tensor_encoding1.per_block_int_scale) == len(tensor_encoding2.per_block_int_scale):
            if not all(
                x == y
                for x, y in zip(
                    tensor_encoding1.per_block_int_scale, tensor_encoding2.per_block_int_scale
                )
            ):
                _info = FieldStatus.PER_BLOCK_INT_SCALE_NOT_SAME.value
                _comparison_status = ComparisonStatus.ERROR
        else:
            _info = FieldStatus.NUM_BLOCKS_NOT_SAME.value.format(
                len(tensor_encoding1.per_block_int_scale), len(tensor_encoding2.per_block_int_scale)
            )
            _comparison_status = ComparisonStatus.ERROR
        info["per_block_int_scale"] = _info
        comparison_status = updated_comparison_status(comparison_status, _comparison_status)

        # ------------------------------------------------------------------------------------------
        # Compare block_size
        # ------------------------------------------------------------------------------------------
        if tensor_encoding1.block_size == tensor_encoding2.block_size:
            info["block_size"] = FieldStatus.SAME.value
        else:
            info["block_size"] = FieldStatus.BLOCK_SIZE_NOT_SAME.value.format(
                tensor_encoding1.block_size, tensor_encoding2.block_size
            )
            comparison_status = updated_comparison_status(comparison_status, ComparisonStatus.ERROR)

        # ------------------------------------------------------------------------------------------
        # Compare compressed_bw
        # ------------------------------------------------------------------------------------------
        if tensor_encoding1.compressed_bw == tensor_encoding2.compressed_bw:
            info["compressed_bw"] = FieldStatus.SAME.value
        else:
            info["compressed_bw"] = FieldStatus.COMPRESSED_BW_NOT_SAME.value.format(
                tensor_encoding1.compressed_bw, tensor_encoding2.compressed_bw
            )
            comparison_status = updated_comparison_status(comparison_status, ComparisonStatus.ERROR)

        return info, comparison_status.value

    def _compare_tensor_encodings(
        self,
        tensor_encoding1: TensorEncoding,
        tensor_encoding2: TensorEncoding,
        scale_threshold: float = 1e-3,
    ) -> tuple[dict, str]:
        """Compares two tensor encodings
        Args:
            tensor_encoding1 (TensorEncoding): TensorEncoding object for tensor1
            tensor_encoding2 (TensorEncoding): TensorEncoding object for tensor2
            scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

        Returns:
            (dict, str): tuple of:
                1. dictionary of comparison info with keys: dtype, is_symm, channels, scale, offset
                2. ComparisonStatus value of the encodings comparision
        """
        if tensor_encoding1.encoding_type == tensor_encoding2.encoding_type:
            if tensor_encoding1.encoding_type in [
                EncodingType.PER_TENSOR,
                EncodingType.PER_CHANNEL,
                EncodingType.UNDEFINED,
            ]:
                info, comparison_status = self._compare_per_tensor_or_channel_tensor_encodings(
                    tensor_encoding1=tensor_encoding1,
                    tensor_encoding2=tensor_encoding2,
                    scale_threshold=scale_threshold,
                )
            elif tensor_encoding1.encoding_type == EncodingType.LPBQ:
                info, comparison_status = self._compare_lpbq_tensor_encodings(
                    tensor_encoding1=tensor_encoding1,
                    tensor_encoding2=tensor_encoding2,
                    scale_threshold=scale_threshold,
                )
            else:
                raise Exception(
                    f"Tensor Encoding comparison for encoding type: {tensor_encoding1.encoding_type} not supported."
                )
            info["encoding_type"] = FieldStatus.SAME.value

        else:
            comparison_status = ComparisonStatus.ERROR.value
            if tensor_encoding1.tensor_name == tensor_encoding2.tensor_name:
                # If both of them are either of type PER_TENSOR, PER_CHANNEL, UNDEFINED
                # Aim is to salvage whatever information and to prevent comparing
                # LPBQ encoding to conventional encodings
                if tensor_encoding1.encoding_type in [
                    EncodingType.PER_TENSOR,
                    EncodingType.PER_CHANNEL,
                    EncodingType.UNDEFINED,
                ] and tensor_encoding2.encoding_type in [
                    EncodingType.PER_TENSOR,
                    EncodingType.PER_CHANNEL,
                    EncodingType.UNDEFINED,
                ]:
                    info, _ = self._compare_per_tensor_or_channel_tensor_encodings(
                        tensor_encoding1=tensor_encoding1,
                        tensor_encoding2=tensor_encoding2,
                        scale_threshold=scale_threshold,
                    )
                else:
                    info = initialize_comparison_dict()

                info["encoding_type"] = FieldStatus.ENCODING_TYPE_NOT_SAME.value.format(
                    tensor_encoding1.encoding_type.value, tensor_encoding2.encoding_type.value
                )

            else:
                info = None

        return info, comparison_status

    def _compare_block_encodings(
        self,
        block_encoding1: dict[str:TensorEncoding],
        block_encoding2: dict[str:TensorEncoding],
        scale_threshold: float = 1e-3,
    ) -> tuple[dict, dict]:
        """Compares each tensor in encoding1 block against each tensor in encoding2 block.

        Args:
            block_encoding1 (dict): Dictionary of tensor encoding with tensor name as key.
            block_encoding2 (dict): Dictionary of tensor encoding with tensor name as key.
            scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

        Returns:
            (dict, dict): dictionary of comparision between each tensor in encodings1 againts each
                tensor in encodings1 and vice-versa. Two tensors are mapped iff:
                1. They share same algebrically similar encodings. One encoding can be algebrically
                   converted to other and vice-versa;
                2. They share same name.
        """
        encoding1_comparison = get_comparison_structure(block_encoding1.keys())
        encoding2_comparison = get_comparison_structure(block_encoding2.keys())

        for tensor_name1, tensor_encoding1 in block_encoding1.items():
            for tensor_name2, tensor_encoding2 in block_encoding2.items():
                compare_info, comparison_status = self._compare_tensor_encodings(
                    tensor_encoding1=tensor_encoding1,
                    tensor_encoding2=tensor_encoding2,
                    scale_threshold=scale_threshold,
                )
                # Map in the following cases:
                # 1. If tensor_encoding1 == tensor_encoding2
                # 2. If encodings are not same but activation names are same
                if (
                    comparison_status
                    in [ComparisonStatus.SUCCESS.value, ComparisonStatus.WARNING.value]
                    or tensor_name1 == tensor_name2
                ):
                    # Remove TWO_FLOAT_ARE_NOT_SAME from the compare_info
                    if FieldStatus.TWO_FLOAT_ARE_NOT_SAME.value in compare_info["dtype"]:
                        compare_info["dtype"] = FieldStatus.SAME.value
                    encoding1_comparison[tensor_name1]["compare_info"][tensor_name2] = compare_info
                    encoding1_comparison[tensor_name1]["Mapping"].append(tensor_name2)
                    encoding1_comparison[tensor_name1]["Status"][tensor_name2] = comparison_status
                    encoding2_comparison[tensor_name2]["compare_info"][tensor_name1] = compare_info
                    encoding2_comparison[tensor_name2]["Mapping"].append(tensor_name1)
                    encoding2_comparison[tensor_name2]["Status"][tensor_name1] = comparison_status

        # For any tensor which has no mappings, assign the Status as ComparisonStatus.UNMAPPED.value
        for tensor_name in encoding1_comparison:
            if not encoding1_comparison[tensor_name]["Mapping"]:
                encoding1_comparison[tensor_name]["Status"] = ComparisonStatus.UNMAPPED.value

        for tensor_name in encoding2_comparison:
            if not encoding2_comparison[tensor_name]["Mapping"]:
                encoding2_comparison[tensor_name]["Status"] = ComparisonStatus.UNMAPPED.value

        return encoding1_comparison, encoding2_comparison

    def _compare(
        self,
        model_encoding1: Encoding,
        model_encoding2: Encoding,
        output_dir: Path,
        scale_threshold: float = 1e-3,
    ) -> None:
        """Compares primary_encodings against the reference encodings.

        Args:
            model_encoding1 (Encoding): Encoding class object for Aimet or Qairt encoding
            model_encoding2 (Encoding): Encoding class object for Aimet or Qairt encoding
            output_dir(Path): path to output directory where analysis will be dumped
            scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3
        """
        if (
            list(model_encoding1.tensor_encodings.values())[0].tensor_type == TensorType.Encodings
            or list(model_encoding2.tensor_encodings.values())[0].tensor_type
            == TensorType.Encodings
        ):
            tensor_types = [TensorType.Encodings]
        else:
            tensor_types = [TensorType.ActivationEncodings, TensorType.ParamEncodings]

        for tensor_type in tensor_types:
            self._logger.info(f"Starting {tensor_type.value} comparision.")
            tensor_encodings_comparison1, tensor_encodings_comparison2 = (
                self._compare_block_encodings(
                    block_encoding1=model_encoding1.get_type_encodings(tensor_type=tensor_type),
                    block_encoding2=model_encoding2.get_type_encodings(tensor_type=tensor_type),
                    scale_threshold=scale_threshold,
                )
            )
            self._logger.info(f"{tensor_type.value} comprision done.")

            # Dump analysis
            self._logger.info(f"Dumping CSV analysis for {tensor_type.value} comparision.")

            _tensor_type = tensor_type.value.split("_")[0]  # param_encodings => param
            # Dump tensor_encodings_comparison1
            tensor_encodings_comparison1_path = output_dir.joinpath(
                f"{model_encoding1.name}_{_tensor_type}.json"
            )
            dump_json(tensor_encodings_comparison1, tensor_encodings_comparison1_path)

            # Dump tensor_encodings_comparison2
            tensor_encodings_comparison2_path = output_dir.joinpath(
                f"{model_encoding2.name}_{_tensor_type}.json"
            )
            dump_json(tensor_encodings_comparison2, tensor_encodings_comparison2_path)

            # Create and dump comparison in the form of csv analysis
            csv_path = output_dir.joinpath(f"{_tensor_type}_comparison.csv")
            self._dump_csv_analysis(
                comparison1=tensor_encodings_comparison1,
                comparison2=tensor_encodings_comparison2,
                csv_path=csv_path,
                filename1=model_encoding1.name,
                filename2=model_encoding2.name,
            )
            self._logger.info(f"CSV analysis for {tensor_type.value} comparison dumped.")

    def _dump_csv_analysis(
        self, comparison1: dict, comparison2: dict, csv_path: Path, filename1: str, filename2: str
    ) -> None:
        """Builds and dumps csv dataframe for the given comparision info
        Args:
            comparison1 (dict): comparision info of comparing encoding1 against encoding2
            comparison2 (dict): comparision info of comparing encoding2 against encoding1
            csv_path (Path): path to the csv file
            filename1 (str): encoding1 file name
            filename2 (str): encoding2 file name
        """

        def _add_compare_info(data_frame: dict, compare_info: dict) -> dict:
            """Add compare_info to data_frame

            Args:
                data_frame (dict): running data frame for csv analysis
                compare_info (dict): dictionary that contains comparison info for a pair of tensor

            Return:
                (dict): updated data_frame dictionary that contains compare_info
            """
            data_frame["encoding_type"].append(compare_info["encoding_type"])
            data_frame["dtype"].append(compare_info["dtype"])
            data_frame["is_symm"].append(compare_info["is_symm"])
            data_frame["bitwidth"].append(compare_info["bitwidth"])
            data_frame["channels"].append(compare_info["channels"])
            data_frame["scale"].append(compare_info["scale"])
            data_frame["offset"].append(compare_info["offset"])
            data_frame["block_size"].append(compare_info["block_size"])
            data_frame["compressed_bw"].append(compare_info["compressed_bw"])
            data_frame["per_block_int_scale"].append(compare_info["per_block_int_scale"])

            return data_frame

        # Initialize the dataframe
        # Tensor Name column includes the filename of the encoding file for better readability
        # If both filenames are same, append an index
        if filename1 == filename2:
            filename1 = filename1 + "_1"
            filename2 = filename2 + "_2"
        tensor_name1 = f"Tensor Name({filename1})"
        tensor_name2 = f"Tensor Name({filename2})"
        columns = [
            tensor_name1,
            tensor_name2,
            "Status",
            "encoding_type",
            "dtype",
            "is_symm",
            "bitwidth",
            "channels",
            "scale",
            "offset",
            "block_size",
            "compressed_bw",
            "per_block_int_scale",
            "Total Mappings",
        ]
        data_frame = {column: [] for column in columns}

        visited_tensors = []
        # Populate the dataframe with tensors present in comparison1
        for tensor_name, tensor_comparison_info in comparison1.items():
            visited_tensors.append(tensor_name)
            data_frame[tensor_name1].append(tensor_name)
            mappings = tensor_comparison_info["Mapping"]
            data_frame["Total Mappings"].append(len(mappings))
            # Three possibilities:
            # 1. tensor_name in comparison1 is not mapped to any of the tensor in comparison2
            # 2. tensor_name is mapped to tensors in comparison2 and also present in the mapping
            # 3. tensor_name is mapped to tensors in comparison2 but not present in the mapping
            #    then pick any of the mapped tensors for csv file
            if not mappings:
                data_frame[tensor_name2].append(None)
                data_frame["Status"].append(tensor_comparison_info["Status"])
                data_frame = _add_compare_info(data_frame, initialize_comparison_dict())

            elif tensor_name in mappings:
                data_frame[tensor_name2].append(tensor_name)
                data_frame["Status"].append(tensor_comparison_info["Status"][tensor_name])
                data_frame = _add_compare_info(
                    data_frame, tensor_comparison_info["compare_info"][tensor_name]
                )
            else:
                data_frame[tensor_name2].append(mappings[0])
                data_frame["Status"].append(tensor_comparison_info["Status"][mappings[0]])
                data_frame = _add_compare_info(
                    data_frame, tensor_comparison_info["compare_info"][mappings[0]]
                )

        # Populate the dataframe with tensors which are not present in dataframe but
        # present in comparison2
        for tensor_name, tensor_comparison_info in comparison2.items():
            if tensor_name not in visited_tensors:
                data_frame[tensor_name2].append(tensor_name)
                mappings = tensor_comparison_info["Mapping"]
                data_frame["Total Mappings"].append(len(mappings))
                # Three possibilities:
                # 1. tensor_name in comparison2 is not mapped to any of the tensor in comparison1
                # 2. tensor_name is mapped to tensors in comparison1 and also present in the mapping
                # 3. tensor_name is mapped to tensors in comparison1 but not present in the mapping
                #    then pick any of the mapped tensors for csv file
                if not mappings:
                    data_frame[tensor_name1].append(None)
                    data_frame["Status"].append(tensor_comparison_info["Status"])
                    data_frame = _add_compare_info(data_frame, initialize_comparison_dict())
                elif tensor_name in mappings:
                    data_frame[tensor_name1].append(tensor_name)
                    data_frame["Status"].append(tensor_comparison_info["Status"][tensor_name])
                    data_frame = _add_compare_info(
                        data_frame, tensor_comparison_info["compare_info"][tensor_name]
                    )
                else:
                    data_frame[tensor_name1].append(mappings[0])
                    data_frame["Status"].append(tensor_comparison_info["Status"][mappings[0]])
                    data_frame = _add_compare_info(
                        data_frame, tensor_comparison_info["compare_info"][mappings[0]]
                    )

        dump_csv(data_frame=data_frame, csv_path=csv_path)

    def _map_supergroup(
        self,
        encodings_converter: QairtEncodingsConverter,
        model_encoding: Encoding,
        output_dir: Path,
    ) -> None:
        """Map each activation in reference encodings to a supergroup in qairt_encodings.
        Dumps the mapping in output directory.

        Args:
            encodings_converter (QairtEncodingsConverter): encodings converter object for
                created with qairt_encodings
            model_encoding (Encoding): Encoding for the model.
            output_dir(Path): path to the output_dir where supergroup analysis will be dumped
        """
        file_path = output_dir.joinpath(f"{model_encoding.name}_supergroup.json")

        # Get the target(Qairt) supergroups
        framework_activation_op_map = encodings_converter.get_framework_activation_op_map()
        target_activation_op_map = encodings_converter.get_target_activation_op_map()

        # Get conv, bn, relu supergroups
        conv_bn_relu_activations = get_supergroup_activations(
            framework_activation_op_map=framework_activation_op_map,
            target_activation_op_map=target_activation_op_map,
        )
        # Get model inputs: ops with no parent ops
        model_inputs = set()
        for activation, target_op in target_activation_op_map.items():
            if not target_op.parent_ops:
                model_inputs.update([activation])

        # target_activations = all target activation
        # (minus) conv_bn_relu_activations
        # (minus) model_inputs
        # which are also present in framework model
        target_activations = (
            target_activation_op_map.keys() - conv_bn_relu_activations
        ) - model_inputs
        target_activations = [
            activation
            for activation in target_activations
            if activation in framework_activation_op_map
        ]

        supergroup_target_activation_map = {}
        target_activation_supergroup_map = {}
        # Map each target activation to a supergroup and vice-versa
        for idx, target_activation in enumerate(target_activations):
            target_op = target_activation_op_map[target_activation]
            parent_activations = set()

            # Get all supergroup inputs
            for op_input in target_op.inputs:
                if op_input in target_activation_op_map:
                    partial_parent_activations = get_common_parent_activations(
                        current_activation=op_input,
                        graph1_activation_op_map=target_activation_op_map,
                        graph2_activation_op_map=framework_activation_op_map,
                        ignore_activations=conv_bn_relu_activations,
                    )
                    parent_activations.update(partial_parent_activations)

            # Get target supergroup activations, inputs, and outputs
            target_supergroup_activations, target_supergroup_inputs, target_supergroup_outputs = (
                get_subgraph(
                    subgraph_input_names=parent_activations,
                    subgraph_output_names=target_op.outputs,
                    graph1_activation_op_map=target_activation_op_map,
                    graph2_activation_op_map=framework_activation_op_map,
                )
            )

            # Get framework supergroup activation
            (
                framework_supergroup_activations,
                _,
                _,
            ) = get_subgraph(
                subgraph_input_names=parent_activations,
                subgraph_output_names=target_op.outputs,
                graph1_activation_op_map=framework_activation_op_map,
            )
            if not framework_supergroup_activations:
                framework_supergroup_activations = [
                    "Due to converter optimizations, framework subgraph is not found"
                ]

            supergroup_name = f"supergroup_{idx}"
            supergroup_target_activation_map[supergroup_name] = {
                "Inputs": ",".join(sorted(target_supergroup_inputs)),
                "Outputs": ",".join(sorted(target_supergroup_outputs)),
                "Qairt Tensors": ",".join(sorted(target_supergroup_activations)),
                "Framework Tensors": ",".join(sorted(framework_supergroup_activations)),
            }
            for supergroup_activation in target_supergroup_activations:
                target_activation_supergroup_map[supergroup_activation] = supergroup_name

        # Map each activation in encodings to some supergroup in target graph
        supergroup_mapping = {}
        for activation in model_encoding.get_type_encodings(
            tensor_type=TensorType.ActivationEncodings
        ).keys():
            if activation in target_activation_supergroup_map:
                supergroup_name = target_activation_supergroup_map[activation]
                supergroup_mapping[activation] = supergroup_target_activation_map[supergroup_name]
            else:
                supergroup_mapping[activation] = "Not Mapped"

        dump_json(supergroup_mapping, file_path)
