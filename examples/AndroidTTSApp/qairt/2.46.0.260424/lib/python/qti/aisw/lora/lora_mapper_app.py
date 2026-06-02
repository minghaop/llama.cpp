# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
import yaml
import os
from dataclasses import dataclass
from qti.aisw.lora.helpers import is_attach_point, get_attach_points
from qti.aisw.converters.common.utils.converter_utils import log_info, log_debug

@dataclass
class LoraMapperAppConfig(object):
    """
    Dataclass for qairt-lora-mapper app config.
    Config can come from command line args or elsewhere.
    """
    lora_config: str
    output_dir: str
    debug: bool = False


def get_pytorch_to_onnx_mapping(app_config, lora_config):
    pytorch_to_onnx_file = lora_config['attach_point_onnx_mapping']
    if not os.path.isabs(pytorch_to_onnx_file):
        base_path = os.path.dirname(app_config.lora_config)
        pytorch_to_onnx_file = os.path.join(base_path, pytorch_to_onnx_file)

    with open(pytorch_to_onnx_file) as file:
        pytorch_to_onnx = json.load(file)

    return pytorch_to_onnx

def validate_pytorch_to_onnx_mapping(pytorch_to_onnx, lora_config):
    log_info("Validating pytorch-to-onnx mapping...")
    pytorch_to_onnx_file = lora_config['attach_point_onnx_mapping']
    seen_onnx_names = set()
    for pytorch_module_name, onnx_names in pytorch_to_onnx.items():
        if len(onnx_names) == 0:
            raise ValueError("{} has key, {}, with no onnx mapping".format(pytorch_to_onnx_file, pytorch_module_name))

        # validate each onnx name maps to one pytorch name
        for onnx_name in onnx_names:
            if onnx_name in seen_onnx_names:
                raise ValueError("{} has repeated onnx name, {}.".format(pytorch_to_onnx_file, onnx_name))
            seen_onnx_names.add(onnx_name)
    log_info("Done validating pytorch-to-onnx mapping. ")

def get_lora_config(app_config):
    with open(app_config.lora_config) as file:
        config = yaml.safe_load(file)

    return config

def get_adapter_configs(app_config, lora_config):
    def get_adapter_config_paths(app_config, lora_config):
        adapter_names = set()
        absolute_paths = list()
        for adapter_info in lora_config['adapter']:
            if adapter_info['name'] in adapter_names:
                raise ValueError("Invalid LoRA YAML : adapter name, {}, is not unique".format(adapter_info['name']))
            adapter_names.add(adapter_info['name'])

            path = adapter_info["lora_config"]
            if os.path.isabs(path):
                absolute_paths.append(path)
            else:
                base_path = os.path.dirname(app_config.lora_config)
                absolute_path = os.path.join(base_path, path)
                absolute_paths.append(absolute_path)
        return absolute_paths

    def validate_adapter_configs(adapter_path, adapter_config):
        log_info("Validating adapter config: {}".format(adapter_path))
        expected_keys = set(['name', 'rank', 'target_modules'])
        if not expected_keys.issubset(set(adapter_config.keys())):
            raise ValueError("Adapter config with name, {}, should have the keys: {}".format(adapter_config['name'], expected_keys))
        log_info("Done validating adapter config.")

    config_paths = get_adapter_config_paths(app_config, lora_config)
    adapter_configs = dict()

    for path in config_paths:
        with open(path) as file:
            adapter_config = json.load(file)
            validate_adapter_configs(path, adapter_config)
            adapter_configs[adapter_config['name']] = adapter_config

    return adapter_configs

def get_adapter_attach_points(adapter_configs, pytorch_to_onnx_map):
    adapter_attach_points = list()
    pytorch_modules = list(pytorch_to_onnx_map.keys())
    for adapter_name, adapter_config in adapter_configs.items():
        log_debug(f"Finding attach points for adapter {adapter_name}...")
        adapter_target_modules = adapter_config.pop("target_modules")
        target_module_to_attach_points = {target_module:[] for target_module in adapter_target_modules}

        for target_module in adapter_target_modules:
            pytorch_attach_points = get_attach_points(target_module, pytorch_modules)
            for pytorch_attach_point in pytorch_attach_points:
                if len(pytorch_to_onnx_map[pytorch_attach_point]) != 1:
                    raise ValueError(
                        "Each target module should map to 1 onnx name. {} maps to {} onnx names." \
                        .format(pytorch_attach_point, len(pytorch_to_onnx_map[pytorch_attach_point])))
                onnx_attach_point = pytorch_to_onnx_map[pytorch_attach_point][0]
                target_module_to_attach_points[target_module].append(onnx_attach_point)

            log_debug("Found {num_attach_points} attach points for target module {target_module} of adapter {adapter_name}.".format(
                num_attach_points=len(target_module_to_attach_points[target_module]),
                target_module=target_module,
                adapter_name=adapter_name))

        log_debug(f"Done finding attach points for adapter {adapter_name}")
        adapter_attach_points.append({adapter_name:{"target_modules":target_module_to_attach_points}})
    return adapter_attach_points


def dump_adapter_attach_points(app_config, adapter_attach_points):
    formatted_adapter_attach_points = {"adapter":adapter_attach_points}
    filename = 'lora_mapper_debug_info.json'
    save_path = os.path.join(app_config.output_dir, filename)

    with open(save_path, 'w') as f:
        json.dump(formatted_adapter_attach_points, f, indent=4)
    log_info("Adapter attach points saved at {}".format(save_path))


def update_adapter_configs(app_config, adapter_configs, adapter_attach_points):
    for adapter_info in adapter_attach_points:
        adapter_name = list(adapter_info.keys())[0]
        target_operator_names = []
        for target_module, onnx_attach_points in adapter_info[adapter_name]["target_modules"].items():
            target_operator_names.extend(onnx_attach_points)

        if len(target_operator_names) == 0:
            raise ValueError(f"Adapter, {adapter_name}, target modules do not "
                             "map to any onnx attach points.")

        adapter_configs[adapter_name]["target_operator_names"] = target_operator_names
    return adapter_configs


def save_output_files(app_config, updated_adapter_configs, lora_config):
    def get_new_file_name(path):
        file_name = os.path.basename(path)
        base_name, extension = os.path.splitext(file_name)
        new_file_name = base_name + "_updated" + extension
        return new_file_name

    def save_updated_adapter_config(adapter_info, updated_adapter_configs, output_dir, lora_config_path):
        adapter_name = adapter_info['name']
        adapter_path = adapter_info['lora_config']
        adapter_config = updated_adapter_configs[adapter_name]

        new_file_name = get_new_file_name(adapter_path)
        new_adapter_path = os.path.join(output_dir, new_file_name)

        with open(new_adapter_path, "w") as f:
            json.dump(adapter_config, f, indent=4)
        return new_adapter_path

    def save_updated_lora_config(lora_config, lora_config_path, output_dir):
        def make_all_paths_absolute(lora_config, lora_config_path):
            def make_path_absolute(path, lora_config_path):
                if path and not os.path.isabs(path):
                    lora_config_directory = os.path.dirname(os.path.abspath(lora_config_path))
                    path = os.path.join(lora_config_directory, path)
                return path

            lora_config['attach_point_onnx_mapping'] = make_path_absolute(lora_config['attach_point_onnx_mapping'], lora_config_path)
            use_case_names = set()
            for use_case_info in lora_config['use-case']:
                if use_case_info["name"] in use_case_names:
                    raise ValueError("Invalid LoRA YAML : use-case name, {}, is not unique".format(use_case_info["name"]))
                use_case_names.add(use_case_info["name"])

                use_case_info['model_name'] = make_path_absolute(use_case_info['model_name'], lora_config_path)

                # quant_overrides and quant_updatable_tensors are optional for the lora config yaml
                if 'quant_overrides' in use_case_info:
                    use_case_info['quant_overrides'] = make_path_absolute(use_case_info['quant_overrides'], lora_config_path)
                if 'quant_updatable_tensors' in use_case_info:
                    use_case_info['quant_updatable_tensors'] = make_path_absolute(use_case_info['quant_updatable_tensors'], lora_config_path)

        new_file_name = get_new_file_name(lora_config_path)
        new_lora_config_path = os.path.join(output_dir, new_file_name)
        make_all_paths_absolute(lora_config, lora_config_path)

        with open(new_lora_config_path, 'w') as f:
            yaml.dump(lora_config, f)
        return new_lora_config_path


    for adapter_info in lora_config['adapter']:
        new_adapter_path = save_updated_adapter_config(adapter_info, updated_adapter_configs, app_config.output_dir, app_config.lora_config)
        adapter_info['lora_config'] = new_adapter_path
        log_info("New adapter config saved at " + new_adapter_path)

    new_lora_config_path = save_updated_lora_config(lora_config, app_config.lora_config, app_config.output_dir)
    log_info("New lora config saved at " + new_lora_config_path)


def resolve_attach_point_name(app_config):
    """
    Creates new adapter config files with onnx operator names and new lora onnx
    config file with these new adapter config files.

    Parameters:
        app_config (object):
            Dataclass containing parameter values.

    Returns:
        None
    """

    log_debug("Getting lora config file...")
    lora_config = get_lora_config(app_config)
    log_debug("Got lora config file.")

    log_debug("Getting pytorch-to-onnx mapping...")
    pytorch_to_onnx_map = get_pytorch_to_onnx_mapping(app_config, lora_config)
    validate_pytorch_to_onnx_mapping(pytorch_to_onnx_map, lora_config)
    log_debug("Got pytorch-to-onnx mapping.")

    log_debug("Getting adapter configs...")
    adapter_configs = get_adapter_configs(app_config, lora_config)
    log_debug("Got adapter configs.")

    adapter_attach_points = get_adapter_attach_points(adapter_configs, pytorch_to_onnx_map)

    if app_config.debug:
        dump_adapter_attach_points(app_config, adapter_attach_points)

    log_debug("Updating adapter configs...")
    updated_adapter_configs = update_adapter_configs(app_config, adapter_configs, adapter_attach_points)
    log_debug("Updated adapter configs.")

    log_debug("Saving output files...")
    save_output_files(app_config, updated_adapter_configs, lora_config)
    log_debug("All files successfully saved.")

    log_info("Mapper completed successfully.")
