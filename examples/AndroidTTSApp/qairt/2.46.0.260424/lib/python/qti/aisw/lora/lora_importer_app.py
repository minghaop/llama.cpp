#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

import os
import yaml
import argparse
import json

# import numpy before qti.aisw.converters.xxxx modules
import numpy as np
import onnx

# Common Imports
from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.utils import validation_utils
from qti.aisw.converters.common.utils.converter_utils import log_error, log_info, log_warning, log_debug
from qti.aisw.converters.common.converter_ir.op_graph_optimizations import IROptimizations
from qti.aisw.converters.qnn_backend.ir_to_dlc import DLCBackend
from qti.aisw.converters.qnn_backend.custom_ops.op_factory import QnnCustomOpFactory
from qti.aisw.converters.common import encodings_json_serializer
from qti.aisw.converters import onnx as onnx_converter
from safetensors.numpy import load_file, save_file
from qti.aisw.dlc_utils.snpe_dlc_utils import ModelInfo
from qti.aisw.converters.common.dlc_quantizer import DLCQuantizer
from qti.aisw.converters.common.backend_awareness import BackendInfo
from qti.aisw.converters.common.utils.io_utils import get_default_output_directory
from qti.aisw.converters.common.graph_property_setter import GraphPropertySetter
try:
    from qti.aisw.converters.common import modeltools
except ImportError as ie:
    from qti.aisw.dlc_utils import modeltools

from qti.aisw.lora.helpers import apply_safetensors_to_onnx

from qti.aisw.converters.common.graph_optimizer import GraphOptimizer, OptimizationStage

module_lora_flow = False

def get_quantizer_args_from_dlc(model_info):
    quantizer_cmd_str = model_info.read_quantizer_command()
    quantizer_args_namespace = {}
    if quantizer_cmd_str != 'N/A':
        if ';' not in quantizer_cmd_str:
            raise RuntimeError("Commandline parsing is not supported on older DLC."
                               " This support has been added since release v2.25."
                               " Please use SDK v2.25 or later SDK for this functionality")

        quantizer_args_to_value_dict = model_info.parse_quantizer_command()
        quantizer_args_namespace = convert_dict_to_namespace(quantizer_args_to_value_dict)
    return quantizer_args_namespace

def get_converter_args_from_dlc(model_info):
    if ';' not in model_info.read_converter_command():
        raise RuntimeError("Commandline parsing is not supported on older DLC."
                           " This support has been added since release v2.25."
                           " Please use SDK v2.25 or later SDK for this functionality")

    converter_args_to_value_dict = model_info.parse_converter_command()
    converter_args_namespace = convert_dict_to_namespace(converter_args_to_value_dict)

    return converter_args_namespace

def get_default_quantizer_args_from_dlc(model_info):
    converter_args_to_value_dict = model_info.parse_quantizer_command()
    converter_args_namespace = convert_dict_to_namespace(converter_args_to_value_dict)
    return converter_args_namespace

def convert_dict_to_namespace(dictionary):
    namespace = argparse.Namespace()
    for key, value in dictionary.items():
        setattr(namespace, key, value)
    return namespace

def get_lora_use_cases(args):
    with open(args.lora_config) as f:
        lora_configs = yaml.safe_load(f)
        return lora_configs['use_case']

def get_safetensor_tensor_names(args):
    lora_use_cases = get_lora_use_cases(args)
    # all use cases should have the same set of updatable tensors
    weights_path = lora_use_cases[0]['lora_weights']
    weights = load_file(weights_path)
    tensor_names = set(weights.keys())
    return tensor_names

def extract_artifacts_from_dlc(dlc_path, quant_updatable_mode):
    dlc_reader = modeltools.IrDlcReader()
    dlc_reader.open(dlc_path)
    graph_names = list(dlc_reader.get_ir_graph_names())
    cpp_graph = dlc_reader.get_ir_graph(graph_names[0])
    converter_command = dlc_reader.converter_command()
    quantizer_command = dlc_reader.quantizer_command()

    lora_metadata_record = {}
    if quant_updatable_mode == "none":
        lora_meta_data_pyrecord = dlc_reader.extract_record("lora.converter.metadata", modeltools.DlcRecordType.LORA_CONVERTER_METADATA)
        lora_metadata_string_record = ''.join(chr(code) for code in lora_meta_data_pyrecord.get_bytes())
        # Convert the lora_metadata string record to dict
        lora_metadata_record = json.loads(lora_metadata_string_record)

    return cpp_graph, converter_command, quantizer_command, lora_metadata_record, dlc_reader

def get_updateable_static_tensor_names_in_graph(cpp_graph):
    updateable_tensor_names = list()
    tensor_map = cpp_graph.get_tensor_map()
    for tensor_name, tensor in tensor_map.items():
        if tensor.is_updateable() and tensor.is_static_tensor():
            updateable_tensor_names.append(tensor_name)
    return updateable_tensor_names

def get_updateable_tensor_names_in_graph(cpp_graph):
    updateable_tensor_names = list()
    tensor_map = cpp_graph.get_tensor_map()
    for tensor_name, tensor in tensor_map.items():
        if tensor.is_updateable():
            updateable_tensor_names.append(tensor_name)
    return updateable_tensor_names

def load_static_tensors_into_graph(tensor_names_to_data_map, py_graph):
    for tensor_name, tensor_data in tensor_names_to_data_map.items():
        if py_graph.has_buffer(tensor_name):
            node = py_graph.get_node_by_name(tensor_name)
            node.op.set_tensor_data(tensor_data)

def get_static_tensor_values(safetensor_tensor_names, cpp_graph):
    tensor_values = dict()
    for tensor_name in safetensor_tensor_names:
        tensor = cpp_graph.get_tensor(tensor_name)
        if tensor.is_static_tensor():
            tensor_values[tensor_name] = tensor.get_data()
    return tensor_values

def get_converter_args_for_lora_conversion(args, converter_args_namespace):
    setattr(converter_args_namespace, "input_network", args.input_network)
    setattr(converter_args_namespace, "debug", args.debug)
    # dummy output path needed for dlc backend
    setattr(converter_args_namespace, "output_path", None)
    # setting transforms metadata explicitly to None
    setattr(converter_args_namespace, "transforms_metadata", None)

    return converter_args_namespace

def get_quantizer_args_for_lora_quantization(args, quantizer_args_namespace):
    setattr(quantizer_args_namespace, "input_dlc", args.input_dlc)
    setattr(quantizer_args_namespace, "output_dlc", None)
    setattr(quantizer_args_namespace, "debug", args.debug)

    if not args.input_list and args.float_fallback:
        setattr(quantizer_args_namespace, "float_fallback", args.float_fallback)

    if args.input_list:
        if args.float_fallback:
            raise Exception("Quantizer invoked with --enable_float_fallback but lora-importer invoked with --input_list")
        setattr(quantizer_args_namespace, "input_list", args.input_list)

    return quantizer_args_namespace

def get_quantizer_args_for_lora_quantization_using_converter_args(args, quantizer_args_namespace):
    setattr(quantizer_args_namespace, "input_dlc", args.input_dlc)
    setattr(quantizer_args_namespace, "output_dlc", None)
    setattr(quantizer_args_namespace, "debug", args.debug)
    setattr(quantizer_args_namespace, "input_list", None)
    setattr(quantizer_args_namespace, "float_fallback", args.float_fallback)
    setattr(quantizer_args_namespace, "adjust_bias_encoding", False)

    if args.input_list:
        raise Exception("Input DLC is quantized using --enable_float_fallback but lora-importer is invoked with --input_list")

    return quantizer_args_namespace

def get_py_graph(converter_args):
    if module_lora_flow:
        # Create a copy of the arguments to avoid modifying the original
        copied_converter_args = dict(vars(converter_args))
        copied_converter_args['lora_weight_list'] = None
        copied_converter_args['converter_op_package_lib'] = ""
    else:
        copied_converter_args = convert_dict_to_namespace(converter_args.__dict__)
        copied_converter_args.converter_op_package_lib = getattr(copied_converter_args, 'converter_op_package_lib', None) or ""
        copied_converter_args.lora_weight_list = None # we don't need to load this file in qairt-lora-importer,
                                                  # and it may not even exist in qairt-lora-importer stage.

    converter = onnx_converter.OnnxConverterFrontend(copied_converter_args, custom_op_factory=QnnCustomOpFactory())
    py_graph = converter.convert()
    return py_graph

def optimize_py_graph(converter_args, py_graph):

    optimizer = IROptimizations(converter_args)
    optimized_graph = optimizer.optimize(py_graph)
    return optimized_graph

def get_cpp_graph(py_graph, backend):
    backend.prepare_py_graph(py_graph)

    cpp_graph = backend.get_ir_graph(py_graph)
    backend.prepare_cpp_graph(py_graph, cpp_graph)

    return cpp_graph

def get_quantizer(quantizer_args, encoding_version):
    args_dict = DLCQuantizer.ArgParser.validate_and_convert_args(quantizer_args)

    # Backend Awareness
    backend_info_obj = BackendInfo.get_instance(quantizer_args.backend, quantizer_args.soc_model)

    args_dict = DLCQuantizer.ArgParser.validate_and_convert_args(quantizer_args)

    # Backend Awareness
    backend_info_obj = BackendInfo.get_instance(quantizer_args.backend, quantizer_args.soc_model)

    # Set use_quantize_v2 based on encoding version
    use_quantize_v2 = encoding_version == "2.0.0"

    dlc_quantizer = DLCQuantizer(input_dlc=args_dict['input_dlc'],
                                    output_dlc=args_dict['output_dlc'],
                                    input_list=args_dict['input_list'],
                                    float_fallback=args_dict['float_fallback'],
                                    param_quantizer=args_dict['param_quantizer'],
                                    act_quantizer=args_dict['act_quantizer'],
                                    algorithms=args_dict['algorithms'],
                                    bias_bitwidth=args_dict['bias_bitwidth'],
                                    act_bitwidth=args_dict['act_bitwidth'],
                                    weights_bitwidth=args_dict['weights_bitwidth'],
                                    float_bitwidth=args_dict['float_bitwidth'],
                                    float_bias_bitwidth=args_dict['float_bias_bitwidth'],
                                    ignore_encodings=args_dict['ignore_encodings'],
                                    use_per_channel_quantization=args_dict['use_per_channel_quantization'],
                                    use_per_row_quantization=args_dict['use_per_row_quantization'],
                                    use_native_input_files=args_dict['use_native_input_files'],
                                    use_native_output_files=args_dict['use_native_output_files'],
                                    restrict_quantization_steps=args_dict['restrict_quantization_steps'],
                                    use_dynamic_16_bit_weights=args_dict['use_dynamic_16_bit_weights'],
                                    pack_4_bit_weights=args_dict['pack_4_bit_weights'],
                                    keep_weights_quantized=args_dict["keep_weights_quantized"],
                                    adjust_bias_encoding=args_dict["adjust_bias_encoding"],
                                    act_quantizer_calibration=args_dict['act_quantizer_calibration'],
                                    param_quantizer_calibration=args_dict['param_quantizer_calibration'],
                                    act_quantizer_schema=args_dict['act_quantizer_schema'],
                                    param_quantizer_schema=args_dict['param_quantizer_schema'],
                                    percentile_calibration_value=args_dict['percentile_calibration_value'],
                                    use_aimet_quantizer=args_dict['use_aimet_quantizer'],
                                    op_package_lib=args_dict['op_package_lib'],
                                    disable_legacy_quantizer=args_dict['disable_legacy_quantizer'],
                                    dump_encoding_json=args_dict['dump_encoding_json'],
                                    include_data_invariant_ops=args_dict['include_data_invariant_ops'],
                                    aimet_config=args_dict['config_file'],
                                    backend_info_obj=backend_info_obj,
                                    use_quantize_v2=use_quantize_v2
                                    )
    return dlc_quantizer

def get_quantized_graph(quantizer_args, cpp_graph, encoding_version):
    quantizer = get_quantizer(quantizer_args, encoding_version)
    quantizer.set_ir_graph(cpp_graph)
    quantizer.quantize()
    quantized_cpp_graph = quantizer.ir_graph
    return quantized_cpp_graph

def get_safetensor_dict(config):
    safetensor_path = config['lora_weights']
    if safetensor_path:
        tensor_names_to_data_map = load_file(safetensor_path)
        return tensor_names_to_data_map
    else:
        return None


def make_tensors_updateable(cpp_graph, tensor_names):
    for tensor_name in tensor_names:
        if not cpp_graph.has_tensor(tensor_name):
            error_message = "Error: Tensor name, {}, not found in the graph.".format(tensor_name)
            log_error(error_message)
        log_debug("Marking tensor {} as updatable in the graph ".format(tensor_name))
        cpp_tensor = cpp_graph.get_tensor(tensor_name)
        cpp_tensor.set_updateable(True)

def check_float_fallback_applied(converter_args):
    return converter_args.export_format == "DLC_DEFAULT" and converter_args.quantization_overrides


def create_lora_updates(args, converter_args, quantizer_args):
    def get_save_directory(config, args):
        if args.output_dir:
            return args.output_dir
        elif ('output_path' not in config) or (config['output_path'] is None):
            return get_default_output_directory(args.lora_config, "qairt_lora_importer_outputs")
        else:
            save_dir = config['output_path']
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            return save_dir

    def save_encoding(cpp_graph, config):
        encoding_version = get_encoding_version(config['quant_overrides'])
        include_data_invariant_ops = True
        use_case_name = config['name']
        filename = use_case_name + "_lora.encodings"
        save_dir = get_save_directory(config, args)
        save_path = os.path.join(save_dir, filename)
        serializer = encodings_json_serializer.IrEncodingsJsonSerializer(
                save_path,
                include_data_invariant_ops)
        serializer.serialize(cpp_graph, encoding_version)
        encoding_json = serializer.get_graph_json()
        with open(save_path, "w") as f:
            f.write(encoding_json)
            log_info(use_case_name + " encodings JSON saved at: " + save_path)
        return save_path

    def save_updatable_static_tensors(config, cpp_graph, updateable_static_tensor_names):
        use_case_name = config['name']
        filename = use_case_name + "_lora.safetensors"
        save_dir = get_save_directory(config, args)
        save_path = os.path.join(save_dir, filename)

        tensor_names_to_data = dict()
        for tensor_name in updateable_static_tensor_names:
            if not cpp_graph.has_tensor(tensor_name):
                log_warning("Warning: Tensor name, {}, in not found in the DLC graph but not in the converted onnx graph.".format(tensor_name))
            else:
                cpp_tensor = cpp_graph.get_tensor(tensor_name)
                cpp_tensor_dtype = cpp_tensor.data_type()
                tensor_data = cpp_tensor.get_data()

                # get_data() returns uint16 buffer for fp16 datatype
                # Read the uint16 buffer in fp16, then convert to fp32
                if cpp_tensor_dtype == ir_graph.QNN_DATATYPE_FLOAT_16:
                    tensor_data_fp16 = np.frombuffer(tensor_data.tobytes(), dtype=np.float16)
                    tensor_data_fp16 = np.reshape(tensor_data_fp16, tensor_data.shape)
                    tensor_data = tensor_data_fp16.astype(np.float32)

                tensor_names_to_data[tensor_name] = tensor_data

        save_file(tensor_names_to_data, save_path)
        log_info(use_case_name + " safetensors saved at: " + save_path)

        return save_path

    def dump_patched_onnx_model(config, input_network, output_dir):
        safetensors_dict = get_safetensor_dict(config)
        usecase_name = config['name']
        model = onnx.load(input_network, load_external_data=True)

        # Apply the safetensors file and generate the patched Onnx model
        patched_model = apply_safetensors_to_onnx(model, safetensors_dict)
        # Save the patched Onnx model
        path = os.path.join(output_dir, "{}.onnx".format(usecase_name))
        onnx.save(patched_model, path, save_as_external_data=True,
                  all_tensors_to_one_file=True, location=usecase_name+".data")
        log_info("Patched ONNX model for use-case {} saved at {}".format(usecase_name, path))

    def set_quant_overrides(converter_args, usecase_config):
        if "quant_overrides" in usecase_config:
            converter_args.quantization_overrides = usecase_config["quant_overrides"]
        else:
            converter_args.quantization_overrides = None

    def apply_lora_transformations(config, tensor_names_to_data_map, lora_metadata, graph_name):
        use_case_name = config['name']
        filename = use_case_name + "_lora.safetensors"
        save_dir = get_save_directory(config, args)
        save_path = os.path.join(save_dir, filename)

        for tensor_name in tensor_names_to_data_map.keys():
            if tensor_name not in lora_metadata[graph_name]["lora_tensors"]:
                continue
            transformations = lora_metadata[graph_name]["lora_tensors"][tensor_name]["transforms"]

            for transform in transformations:
                if transform["type"] == "Reshape":
                    shape_attribute = transform["tensor_attributes"][0]["shape"]["data"]
                    tensor_names_to_data_map[tensor_name] = tensor_names_to_data_map[tensor_name].reshape(shape_attribute)
                elif transform["type"] == "Transpose":
                    perm_attribute = transform["tensor_attributes"][0]["perm"]["data"]
                    tensor_names_to_data_map[tensor_name] = tensor_names_to_data_map[tensor_name].transpose(perm_attribute)
                elif transform["type"] == "Cast":
                    cast_attribute = transform["scalar_attributes"][0]["to_dtype"]["data"]
                    tensor_names_to_data_map[tensor_name] = tensor_names_to_data_map[tensor_name].astype(cast_attribute)
                else:
                    log_error("Unsupported transformation: {}, found in the LoRA metadata. "
                              "Use --skip_apply_graph_transforms to disable its application".format(transform["type"]))

        save_file(tensor_names_to_data_map, save_path)
        log_info(use_case_name + " safetensors saved at: " + save_path)

        return save_path

    def set_static_data_into_irgraph(cpp_graph, transformed_safetensor_path, lora_tensors):
        tensor_names_data_map = load_file(transformed_safetensor_path)

        for tensor_name, tensor_data in tensor_names_data_map.items():
            if tensor_name not in lora_tensors:
                continue

            if not cpp_graph.has_tensor(tensor_name):
                error_message = "Error: Tensor name, {}, not found in the graph.".format(tensor_name)
                log_error(error_message)

            log_debug("Setting the tensor: {} data in the graph ".format(tensor_name))
            cpp_tensor = cpp_graph.get_tensor(tensor_name)
            cpp_tensor.update_data(tensor_data)

    def get_serialized_encoding_json(cpp_graph, encoding_version):
        include_data_invariant_ops = True
        serializer = encodings_json_serializer.IrEncodingsJsonSerializer('', include_data_invariant_ops)
        serializer.serialize(cpp_graph, encoding_version)
        encoding_json = serializer.get_graph_json()

        return encoding_json

    def save_json_encodings(json_encodings, config):
        use_case_name = config['name']
        filename = use_case_name + "_lora.encodings"
        save_dir = get_save_directory(config, args)
        save_path = os.path.join(save_dir, filename)
        with open(save_path, "w") as f:
            f.write(json_encodings)
            log_info(use_case_name + " encodings JSON saved at: " + save_path)

        return save_path

    def save_dlc(cpp_graph, config, args, converter_command, quantizer_command):
        use_case_name = config['name']
        filename = use_case_name + ".dlc"
        save_dir = get_save_directory(config, args)
        save_path = os.path.join(save_dir, filename)
        dlc_writer = modeltools.IrDlcSerializer(save_path, "", "", converter_command, quantizer_command)
        dlc_writer.initialize()
        dlc_writer.serialize(cpp_graph)
        dlc_writer.finish()
        log_info(use_case_name + " DLC saved at: " + save_path)

    def get_encoding_version(encoding_filepath):
        with open(encoding_filepath, 'r') as file:
            encodings = json.load(file)
        return encodings['version']


    graph_property_setter = GraphPropertySetter()
    quant_updatable_mode = None
    if hasattr(converter_args, "quant_updatable_mode") and converter_args.quant_updatable_mode == "none" and \
        not (hasattr(converter_args, "disable_transform_tracking") and converter_args.disable_transform_tracking):
        quant_updatable_mode = converter_args.quant_updatable_mode

    dlc_graph, converter_command, quantizer_command, lora_metadata, _ = extract_artifacts_from_dlc(args.input_dlc, quant_updatable_mode)
    updateable_static_tensor_names = get_updateable_static_tensor_names_in_graph(dlc_graph)
    updateable_tensor_names = get_updateable_tensor_names_in_graph(dlc_graph)
    lora_configs = get_lora_use_cases(args)
    output_config_data = {"use_case":[]}
    encoding_json = {}
    encoding_version = None

    float_fallback_applied = check_float_fallback_applied(converter_args)
    should_quantize = False
    if float_fallback_applied or ((args.input_list or args.float_fallback) and quantizer_args):
        should_quantize = True

    if should_quantize and quant_updatable_mode == "none" and not args.skip_apply_graph_transforms and \
        not (hasattr(converter_args, "disable_transform_tracking") and converter_args.disable_transform_tracking) and \
        not args.dump_usecase_dlc:
        # Get the serialized json encodings for the quant_updatable_mode = none. The quantization encodings will
        # be same for all lora configs in such case.
        encoding_version = get_encoding_version(lora_configs[0]['quant_overrides'])
        encoding_json = get_serialized_encoding_json(dlc_graph, encoding_version)

    for config in lora_configs:
        if args.dump_usecase_onnx:
            # Dump the patched ONNX model
            output_dir = get_save_directory(config, args)

            # Raise warning if dump_onnx is passed as the command line argument and input_netwrork is not passed in
            # the command line argument. --input_network is not a mandatory argument for quant_updatable_mode = none.
            # If the user wants to dump the patched onnx model then the input_network path must be passed.
            if quant_updatable_mode == "none" and not args.skip_apply_graph_transforms and not args.input_network:
                log_warning("For quant_updatable_mode = none, --input_network is not a mandatory argument. "
                            "To dump patched onnx model, pass the input model path with the --input_network argument")
            else:
                dump_patched_onnx_model(config, args.input_network, output_dir)

        set_quant_overrides(converter_args, config)

        tensor_names_to_data_map = get_safetensor_dict(config)

        # Dictionary to store use case infos like safetensors path, encodings path.
        use_case_info = dict()
        use_case_name = config['name']
        use_case_info['name'] = use_case_name

        # TODO: Add support for dump_usecase_dlc for quant_updatable_mode = "none" in the new flow. If
        # --dump_usecase_dlc argument is passed (it is a suppressed argument) then the old lora-importer
        # flow will be invoked.
        if should_quantize and quant_updatable_mode == "none" and not args.skip_apply_graph_transforms and \
            not (hasattr(converter_args, "disable_transform_tracking") and converter_args.disable_transform_tracking) and \
            not args.dump_usecase_dlc:
            validation_utils.validate_lora_metadata(lora_metadata, dlc_graph.name)

            # Get the transformed lora.safetensors path
            safetensor_save_path = apply_lora_transformations(config, tensor_names_to_data_map, lora_metadata, dlc_graph.name)

            # Update the use_case_info with the saved encodings path. For quant_updatable_mode = none, the encodings
            # will be same for all the lora configs.
            encoding_save_path = save_json_encodings(encoding_json, config)
            use_case_info['encodings'] = encoding_save_path
        else:
            py_graph = get_py_graph(converter_args)

            # Set graph properties originally set by qairt-converter
            graph_property_setter.copy_ir_graph_properties(py_graph, dlc_graph)

            safetensor_path = config['lora_weights']
            validation_utils.validate_tensor_names_in_graph(tensor_names_to_data_map, py_graph, safetensor_path, args.skip_validation)
            if (tensor_names_to_data_map):
                # Safetensors are passed in lora_config
                load_static_tensors_into_graph(tensor_names_to_data_map, py_graph)
            else:
                # Safetensors are not passed in lora_config
                if converter_args.lora_weight_list != "":
                    # error out as qairt-converter was passed with valid lora_weight_list and
                    # qairt-lora-importer should have safetensors files as well
                    raise ValueError(f"No Safetensors passed in lora_config but the Model has lora adapter weights")
                else:
                    log_debug("Model does not have any lora adapter weights and no safetensors are passed in lora_config")

            optimized_py_graph = optimize_py_graph(converter_args, py_graph)

            # Create the DLCBackend to get the prepared cpp graph.
            backend = DLCBackend(converter_args)
            cpp_graph = get_cpp_graph(optimized_py_graph, backend)

            ir_optimizer_args_dict = GraphOptimizer.ArgParser.convert_args(converter_args)
            optimizer = GraphOptimizer(ir_optimizer_args_dict)
            optimizer.optimize(cpp_graph, [OptimizationStage.PostLayout])

            # Get the transformed lora.safetensors path
            safetensor_save_path = save_updatable_static_tensors(config, cpp_graph, updateable_static_tensor_names)

            if should_quantize:
                if cpp_graph.is_adapter_only_quant_updateable():
                    make_tensors_updateable(cpp_graph, updateable_tensor_names)
                # Get encoding version from config for quantization
                if encoding_version is None:
                    encoding_version = get_encoding_version(config['quant_overrides'])
                cpp_graph = get_quantized_graph(quantizer_args, cpp_graph, encoding_version)

                encoding_save_path = save_encoding(cpp_graph, config)
                use_case_info['encodings'] = encoding_save_path

            if args.dump_usecase_dlc:
                # Serialize the cpp_graph to the dlc format
                save_dlc(cpp_graph, config, args, converter_command, quantizer_command)

        use_case_info['weights'] = safetensor_save_path
        use_case_info['graph'] = dlc_graph.name

        output_config_data['use_case'].append(use_case_info)

    save_dir = get_save_directory(config, args)
    save_path = os.path.join(save_dir, "lora_output_files.yaml")

    # Dump the yaml file containing all the relevant use_cases infos.
    with open(save_path, "w") as f:
        yaml.dump(output_config_data, f)


def validate_quantizer_args(lora_args, converter_args, quantizer_args):
    """
                        |       lora_args
                        | --------------------------------------
                        | input_list| float_fallback | None
    quantizer_args    |-----------|----------------|----------
    ------------------|           |                |
    N/A(FP conversion)| Not Valid | Not Valid      | Valid
    input_list        | Valid     | Not Valid      | Not Valid
    float_fallback    | Not Valid | Valid          | Not Valid


    float_fallback can now be applied by default through QAIRT Converter as well, if Encodings are provided/present
    """
    float_fallback_applied = check_float_fallback_applied(converter_args)
    if quantizer_args.input_list and not lora_args.input_list:
        raise Exception("No --input_list has been passed to qairt-lora-importer, but input DLC was quantized using "
                        "--input_list. Please retry after providing calibration data using --input_list")
    elif (quantizer_args.float_fallback or float_fallback_applied) and (not lora_args.input_list and not lora_args.float_fallback):
        raise Exception("No --input_list or --enable_float_fallback has been passed to qairt-lora-importer, "
                        "but input DLC was quantized using --enable_float_fallback. Please retry after "
                        "providing --enable_float_fallback")

    if (quantizer_args.input_list and lora_args.input_list) and (quantizer_args.input_list != lora_args.input_list):
        log_warning("Input list path used to produce the DLC is different than the provided input list path. "
                    "Using the provided input list path to produce lora artifacts.")


def validate_converter_quantized_args(lora_args):
    """
                        |       lora_args
                        | --------------------------------------
                        | input_list| float_fallback | None
    converter_args    |-----------|----------------|----------
    ------------------|           |                |
    N/A(FP conversion)| Not Valid | Not Valid      | Valid
    float_fallback    | Not Valid | Valid          | Not Valid


    float_fallback can now be applied by default through QAIRT Converter as well, if Encodings are provided/present
    """
    if lora_args.input_list:
        raise Exception("Input DLC was quantized using --enable_float_fallback but input list path is provided "
                            "to produce lora artifacts. Please retry after providing --enable_float_fallback")

def apply_lora_updates(args):
    model_info = ModelInfo(args.input_dlc)
    converter_args = get_converter_args_from_dlc(model_info)
    input_network_required = True
    if hasattr(converter_args, "quant_updatable_mode") and converter_args.quant_updatable_mode == "none" and \
        not (hasattr(converter_args, "disable_transform_tracking") and converter_args.disable_transform_tracking) and \
        not args.skip_apply_graph_transforms:
        input_network_required = False

    if input_network_required and not args.input_network:
        if args.skip_apply_graph_transforms:
            log_warning("--skip_apply_graph_transforms can only be used with quant_updatable_mode = none")
        raise Exception("--input_network is required for quant_updatable_mode other than none")

    quantizer_args = get_quantizer_args_from_dlc(model_info)
    float_fallback_applied = check_float_fallback_applied(converter_args)
    converter_args = get_converter_args_for_lora_conversion(args, converter_args)

    if quantizer_args:
        validate_quantizer_args(args, converter_args, quantizer_args)
        quantizer_args = get_quantizer_args_for_lora_quantization(args, quantizer_args)

    elif float_fallback_applied:
        backend = DLCBackend(converter_args)
        validate_converter_quantized_args(args)
        converter_quantized_default_args = get_default_quantizer_args_from_dlc(model_info)
        backend_info_obj = BackendInfo.get_instance(converter_args.backend, converter_args.soc_model)
        converter_float_fallback_default_args = backend.set_qairt_converter_float_fallback_default_args(converter_quantized_default_args, backend_info_obj)
        quantizer_args = get_quantizer_args_for_lora_quantization_using_converter_args(args, converter_float_fallback_default_args)

    else:
        if args.input_list or args.float_fallback:
            raise Exception("Input DLC is not quantized. --input_list or --enable_float_fallback is not a valid "
                            "argument in this case. Please retry after removing this argument")

    create_lora_updates(args, converter_args, quantizer_args)
