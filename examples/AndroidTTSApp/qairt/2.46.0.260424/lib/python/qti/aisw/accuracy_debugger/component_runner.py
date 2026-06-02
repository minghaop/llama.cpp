# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json

import pandas as pd
from qti.aisw.accuracy_debugger.argparser.compare_encodings_parser import CompareEncodingsParser
from qti.aisw.accuracy_debugger.argparser.framework_runner_parser import FrameworkRunnerParser
from qti.aisw.accuracy_debugger.argparser.inference_engine_parser import InferenceEngineParser
from qti.aisw.accuracy_debugger.argparser.model_snooper_parser import ModelSnooperParser
from qti.aisw.accuracy_debugger.argparser.tensor_visualizer_parser import TensorVisualizerParser
from qti.aisw.accuracy_debugger.argparser.validate_encoding_parser import ValidateEncodingParser
from qti.aisw.accuracy_debugger.argparser.verifier_parser import VerifierParser
from qti.aisw.accuracy_debugger.common_config import EncodingInputConfig
from qti.aisw.accuracy_debugger.compare_encodings.compare_encodings import CompareEncodings
from qti.aisw.accuracy_debugger.inference_engine.inference_engine_wrapper import (
    InferenceEngineWrapper,
)
from qti.aisw.accuracy_debugger.inference_engine.qairt_inference_engine import (
    InferenceEngineInputConfig,
)
from qti.aisw.accuracy_debugger.model_snooper.config import (
    BackendConfig,
    ModelSnooperInputConfig,
)
from qti.aisw.accuracy_debugger.model_snooper.module import ModelSnooper
from qti.aisw.accuracy_debugger.tensor_visualizer.tensor_visualizer import TensorVisualizer
from qti.aisw.accuracy_debugger.utils.constants import Algorithm
from qti.aisw.accuracy_debugger.utils.file_utils import dump_csv
from qti.aisw.accuracy_debugger.utils.helper import (
    create_working_directory,
    create_working_directory_with_timestamp,
    get_logger,
    load_input_tensors,
)
from qti.aisw.tools.core.utilities.framework.framework_manager import FrameworkManager
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.tools.core.utilities.verifier.verifier import Verifier


LOG_FILE_NAME = "accuracy_debugger"


def execute_framework_runner(args: list) -> None:
    """Parses arguments for Framework runner and executes it.

    Args:
        args (list): List of arguments to be parsed.
    """
    # Parse arguments for framework runner.
    framework_args = FrameworkRunnerParser().parse(args)
    working_directory = create_working_directory(
        working_directory=framework_args.working_directory,
        sub_directory="framework_runner",
        output_directory=framework_args.output_directory,
    )
    # Get logger for framework runner.
    logger = get_logger(
        logger_name="Framework Runner",
        log_file_path=working_directory,
        log_file_name=LOG_FILE_NAME,
        level=framework_args.log_level.upper(),
    )

    logger.info("Starting framework runner...")
    try:
        # Create framework manager object.
        framework_manager = FrameworkManager(parent_logger=logger)

        # Load input model and input data.
        input_model = framework_manager.load(framework_args.input_model)
        input_data = load_input_tensors(framework_args.input_sample)
        infer_shape = len(input_model.graph.value_info) == 0
        # Generate intermediate outputs.
        reference_outputs = framework_manager.generate_intermediate_outputs(
            input_model,
            input_data,
            output_tensor_names=framework_args.output_tensor,
            infer_shape=infer_shape,
        )
    except Exception:
        logger.exception("Error occurred while running framework runner.")
        raise

    Helper.save_output_to_file(reference_outputs, working_directory)


def execute_inference_engine(args: list) -> None:
    """Parses arguments for Inference engine and executes it.

    Args:
        args (list): Arguments to be provided for inference engine parser.
    """
    # Parse arguments for inference engine.
    inference_args = InferenceEngineParser().parse(args)

    working_directory = create_working_directory(
        working_directory=inference_args.working_directory,
        sub_directory="inference_engine",
        output_directory=inference_args.output_directory,
    )
    # Get logger for Inference Engine.
    logger = get_logger(
        logger_name="Inference Engine",
        log_file_path=working_directory,
        log_file_name=LOG_FILE_NAME,
        level=inference_args.log_level.upper(),
    )
    # Remove log_level argument from inference engine args
    delattr(inference_args, "log_level")
    logger.info("Starting Inference engine...")

    try:
        # Create inference engine object.
        inference_engine = InferenceEngineWrapper(logger=logger)

        # Create inference engine input config with LoRA support
        inference_input_config = InferenceEngineInputConfig(
            input_model=inference_args.input_model,
            converter_arguments=inference_args.converter_args,
            quantizer_arguments=inference_args.quantizer_args,
            backend=inference_args.backend,
            platform=inference_args.platform,
            context_bin_backend_extension=inference_args.offline_prepare_backend_extension_config,
            offline_prepare=inference_args.offline_prepare,
            net_run_arguments=inference_args.net_run_args,
            net_run_input_data=inference_args.input_list,
            net_run_backend_extension=inference_args.netrun_backend_extension_config,
            dump_output=True,
            remote_host_details=inference_args.remote_host_details,
            working_directory=working_directory,
            context_bin_gen_arguments=inference_args.context_bin_args,
            soc_model=inference_args.soc_model,
            # LoRA-specific arguments - now using properly parsed config objects
            lora_model_creator_args=getattr(inference_args, "lora_model_creator_args", None),
            lora_importer_args=getattr(inference_args, "lora_importer_args", None),
            use_case_names=getattr(inference_args, "use_case_names", None),
            lora_alpha_tensor=getattr(inference_args, "lora_alpha_tensor", None),
        )

        # Run inference engine
        output = inference_engine.run(inference_input_config)
    except Exception:
        logger.exception("Error occurred while running inference engine.")
        raise

    logger.info(f"Inference Engine Completed. Results path: {output.output_dir}")


def execute_verification(args: list) -> None:
    """Parses arguments for Verifier and executes it.

    Args:
        args (list): List of arguments to be parsed.
    """
    # Parse arguments for Verifier
    verification_args = VerifierParser().parse(args)

    working_directory = create_working_directory_with_timestamp(
        working_directory=verification_args.working_directory, sub_directory="verification"
    )
    # Get logger for Inference Engine.
    logger = get_logger(
        logger_name="Verifier",
        log_file_path=working_directory,
        log_file_name=LOG_FILE_NAME,
        level=verification_args.log_level.upper(),
    )

    logger.info("Starting Verifier...")
    try:
        if verification_args.graph_info:
            logger.info(f"Loading graph info from {verification_args.graph_info}...")
            with open(verification_args.graph_info, "r") as fp:
                verification_args.graph_info = json.load(fp)

        # Create Verifier object with required comparators
        logger.info(
            "Creating Verifier object with the following comparators: %s",
            ", ".join([comp.name for comp in verification_args.comparators]),
        )
        verifier_obj = Verifier(logger=logger, comparators=verification_args.comparators)

        # Execute verification
        logger.info(
            "Running verifier for the following tensors: \ninference_tensor: %s \nreference_tensor: %s",
            verification_args.inference_tensor,
            verification_args.reference_tensor,
        )
        verifier_output = verifier_obj.verify_directory_of_tensors(
            inference_tensors=verification_args.inference_tensor,
            inference_dtype=verification_args.inference_dtype,
            reference_tensors=verification_args.reference_tensor,
            reference_dtype=verification_args.reference_dtype,
            dlc_file=verification_args.dlc_file,
            graph_info=verification_args.graph_info,
            disable_layout_transform=verification_args.is_qnn_golden_reference,
        )

        verifier_df = pd.DataFrame.from_dict(verifier_output, orient="index")
        verifier_df.index.names = ["inference_tensor_name", "reference_tensor_name"]
        output_csv = working_directory / "verification.csv"

        logger.info("Saving verification results to %s", output_csv)
        dump_csv(data_frame=verifier_df, csv_path=output_csv, index=True)
    except Exception:
        logger.exception("Error occurred while executing verifier.")
        raise

    logger.info("Finished Verifier execution.")


def execute_compare_encodings(args: list) -> None:
    """Parses arguments and runs the Compare encodings.

    Args:
        args (list): List of arguments to be parsed.
    """
    # Parse arguments for Compare encodings
    compare_encodings_args = CompareEncodingsParser().parse(args)

    working_directory = create_working_directory_with_timestamp(
        working_directory=compare_encodings_args.working_directory, sub_directory="compare_encodings"
    )
    # Get logger for Compare Encodings.
    logger = get_logger(
        logger_name="Compare Encodings",
        log_file_path=working_directory,
        log_file_name=LOG_FILE_NAME,
        level=compare_encodings_args.log_level.upper(),
    )
    logger.info("Running Compare Encodings...")

    try:
        # Create EncodingInputConfig
        encoding_config1 = EncodingInputConfig(
            encoding_path=compare_encodings_args.encoding_path1,
            quantized_dlc_path=compare_encodings_args.quantized_dlc1_path,
        )
        encoding_config2 = EncodingInputConfig(
            encoding_path=compare_encodings_args.encoding_path2,
            quantized_dlc_path=compare_encodings_args.quantized_dlc2_path,
        )

        CompareEncodings(logger).run(
            encoding_config1=encoding_config1,
            encoding_config2=encoding_config2,
            output_dir=working_directory,
            framework_model_path=compare_encodings_args.framework_model_path,
            scale_threshold=compare_encodings_args.scale_threshold,
        )

        logger.info(f"Comparison complete. Results saved to: {working_directory}")

    except Exception:
        logger.exception("Error occurred while executing Compare Encodings.")
        raise

    logger.info("Finished Compare Encodings.")


def execute_tensor_visualizer(args: list) -> None:
    """Parses arguments and runs the Tensor Visualizer.

    Args:
        args (list): List of arguments to be parsed.
    """
    # Parse arguments for Tensor Visualizer
    tensor_visualizer_args = TensorVisualizerParser().parse(args)
    working_directory = create_working_directory_with_timestamp(
        working_directory=tensor_visualizer_args.working_directory,
        sub_directory="tensor_visualizer",
    )

    # Create logger for Tensor Visualizer
    logger = get_logger(
        logger_name="Tensor Visualizer",
        log_file_path=working_directory,
        log_file_name=LOG_FILE_NAME,
        level=tensor_visualizer_args.log_level.upper(),
    )
    # Run Tensor Visualizer
    logger.info("Running Tensor Visualizer...")

    try:
        TensorVisualizer(logger).run(
            target_tensors=tensor_visualizer_args.target_tensors,
            golden_tensors=tensor_visualizer_args.golden_tensors,
            working_directory=working_directory,
            datatype=tensor_visualizer_args.data_type,
        )
    except Exception:
        logger.exception("Error occurred while executing Tensor Visualizer.")
        raise
    logger.info("Finished Tensor Visualizer...")


def execute_model_snooper(args: list) -> None:
    """Parses arguments for Model snooper and executes it.

    Args:
        args (list): List of arguments to be parsed.
    """
    # Parse arguments for model snooper
    snooper_args = ModelSnooperParser().parse(args)
    algorithm = getattr(snooper_args, "algorithm", "oneshot")
    working_directory = create_working_directory(
        working_directory=snooper_args.working_directory,
        sub_directory=algorithm + "_snooping",
        output_directory=snooper_args.output_directory
        if hasattr(snooper_args, "output_directory")
        else None,
    )

    # Create Logger using get_logger method for Model Snooper
    logger = get_logger(
        logger_name="Snooping",
        log_file_path=working_directory,
        log_file_name=LOG_FILE_NAME,
        level=snooper_args.log_level.upper(),
    )

    # Remove log_level argument from snooper args
    delattr(snooper_args, "log_level")

    logger.info("Running model snooping...")
    try:
        # Create model snooper input config object
        if hasattr(snooper_args, "target_config"):
            # Config file mode
            reference_config = BackendConfig(**snooper_args.reference_config)
            target_config = BackendConfig(**snooper_args.target_config)

            is_qnn_golden_reference = False
            if reference_config:
                is_qnn_golden_reference = True
            # Convert algorithm string to Algorithm enum if needed
            algorithm_value = getattr(snooper_args, "algorithm", "oneshot")
            if isinstance(algorithm_value, str):
                algorithm_value = Algorithm(algorithm_value)

            snooper_input_config = ModelSnooperInputConfig(
                input_model=getattr(snooper_args, "input_model", None),
                algorithm=algorithm_value,
                reference_config=reference_config,
                target_config=target_config,
                input_sample=snooper_args.input_sample,
                comparators=snooper_args.comparators,
                working_directory=working_directory,
                is_qnn_golden_reference=is_qnn_golden_reference,
                retain_compilation_artifacts=getattr(
                    snooper_args, "retain_compilation_artifacts", False
                ),
                dump_output_tensors=getattr(snooper_args, "dump_output_tensors", True),
                # LoRA-specific arguments for config mode
                lora_model_creator_args=getattr(snooper_args, "lora_model_creator_args", None),
                lora_importer_args=getattr(snooper_args, "lora_importer_args", None),
                use_case_names=getattr(snooper_args, "use_case_names", None),
                lora_alpha_tensor=getattr(snooper_args, "lora_alpha_tensor", None),
            )
        else:
            # CLI mode
            target_config = BackendConfig(
                converter_arguments=snooper_args.converter_args,
                quantizer_arguments=snooper_args.quantizer_args,
                context_bin_gen_arguments=snooper_args.context_bin_args,
                context_bin_backend_extension=snooper_args.offline_prepare_backend_extension_config,
                offline_prepare=snooper_args.offline_prepare,
                net_run_arguments=snooper_args.net_run_args,
                net_run_backend_extension=snooper_args.netrun_backend_extension_config,
                backend=snooper_args.backend,
                platform=snooper_args.platform,
                soc_model=snooper_args.soc_model,
                remote_host_details=snooper_args.remote_host_details,
            )
            snooper_input_config = ModelSnooperInputConfig(
                input_model=snooper_args.input_model,
                reference_config=None,
                target_config=target_config,
                input_sample=snooper_args.input_sample,
                algorithm=snooper_args.algorithm,
                comparators=snooper_args.comparator,
                debug_subgraph_inputs=snooper_args.debug_subgraph_inputs,
                debug_subgraph_outputs=snooper_args.debug_subgraph_outputs,
                skip_layer_types=snooper_args.skip_layer_types,
                include_layer_types=snooper_args.include_layer_types,
                working_directory=working_directory,
                golden_reference_path=snooper_args.golden_reference,
                is_qnn_golden_reference=snooper_args.is_qnn_golden_reference,
                retain_compilation_artifacts=snooper_args.retain_compilation_artifacts,
                dump_output_tensors=True,
                compulsory_overrides=snooper_args.compulsory_overrides,
                max_parallel_compilations=snooper_args.max_parallel_compilations,
                # LoRA-specific arguments
                lora_model_creator_args=getattr(snooper_args, "lora_model_creator_args", None),
                lora_importer_args=getattr(snooper_args, "lora_importer_args", None),
                use_case_names=getattr(snooper_args, "use_case_names", None),
                lora_alpha_tensor=getattr(snooper_args, "lora_alpha_tensor", None),
            )
        # Run Model Snooper
        model_snooper = ModelSnooper(logger)
        output = model_snooper.run(snooper_input_config)
    except Exception:
        logger.exception("Error occurred while executing Snooping.")
        raise
    else:
        logger.info(f"Snooping Completed. CSV Report generated at: {output.csv_snooping_report}")
        logger.info(f"Json Report generated at: {output.json_snooping_report}")


def execute_validate_encoding(args: list) -> None:
    """Parses arguments and runs the Validate Encoding.

    Args:
        args (list): List of arguments to be parsed.
    """
    from qti.aisw.accuracy_debugger.validate_encodings import ValidateEncodings

    # Parse arguments for Validate Encoding
    validate_encoding_args = ValidateEncodingParser().parse(args)

    working_directory = create_working_directory_with_timestamp(
        working_directory=validate_encoding_args.working_directory,
        sub_directory="validate_encoding",
    )
    # Get logger for Validate Encoding.
    logger = get_logger(
        logger_name="Validate Encoding",
        log_file_path=working_directory,
        log_file_name=LOG_FILE_NAME,
        level=validate_encoding_args.log_level.upper(),
    )

    logger.info("Running Validate Encoding...")

    try:
        # Create ValidateEncodings object
        validate_encodings = ValidateEncodings(logger=logger)

        # Run validation
        result = validate_encodings.run(
            encoding_path=validate_encoding_args.encoding_path,
            output_dir=working_directory,
            dlc_file_path=validate_encoding_args.dlc_file_path,
            rule_config_path=validate_encoding_args.rule_config,
        )

        # Log summary
        if result["total_violations"] > 0:
            logger.warning(
                f"Validation completed with {result['total_violations']} violation(s) found."
            )
        else:
            logger.info("✅ All checked tensors satisfy the defined rules.")

        logger.info(f"Validation report saved to: {result['json_report_path']}")
        logger.info(f"CSV report saved to: {result['csv_report_path']}")

    except Exception:
        logger.exception("Error occurred while executing Validate Encoding.")
        raise

    logger.info("Finished Validate Encoding.")