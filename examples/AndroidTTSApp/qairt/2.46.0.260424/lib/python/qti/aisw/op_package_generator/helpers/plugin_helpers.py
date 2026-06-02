# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""
QHPI Helper Functions for QNN Op Package Generator

This module provides helper functions for generating QHPI C code
based on the hexnn_qhpi.h API when UseQHPI is true in the XML configuration.
"""

from qti.aisw.op_package_generator.generator import *


def get_plugin_config_flags(operator, package_info):
    """
    Generate plugin configuration flags based on default settings.

    Args:
        operator: The operator object
        package_info: The package info object

    Returns:
        str: C++ code for plugin configuration structure initialization
    """
    # Auto-detect based on operator definition
    # Check if min_inputs != total inputs or min_outputs != total outputs
    min_inputs = get_min_inputs_from_operator(operator)
    total_inputs = len(operator.input) if operator.input else 0
    min_outputs = get_min_outputs_from_operator(operator)
    total_outputs = len(operator.output) if operator.output else 0

    variable_inputs = (min_inputs != total_inputs)
    variable_outputs = (min_outputs != total_outputs)

    # Default configuration for UseQHPI
    config_lines = []
    config_lines.append(".source_destructive = false,")
    config_lines.append(".self_sliced = false,")
    config_lines.append(f".variable_inputs = {'true' if variable_inputs else 'false'},")
    config_lines.append(f".variable_outputs = {'true' if variable_outputs else 'false'}")
    return "\n    ".join(config_lines)


def get_min_inputs_from_operator(operator):
    """Calculate minimum number of mandatory input tensors from operator"""
    if not operator.input:
        return 0
    # Check if TensorInfo has mandatory attribute, otherwise use all inputs as mandatory
    if hasattr(operator.input[0], 'mandatory'):
        return sum(1 for input_tensor in operator.input if input_tensor.mandatory)
    else:
        # Fallback: assume all inputs are mandatory
        return len(operator.input)


def get_min_outputs_from_operator(operator):
    """Calculate minimum number of mandatory output tensors from operator"""
    if not operator.output:
        return 0
    # Check if TensorInfo has mandatory attribute, otherwise use all outputs as mandatory
    if hasattr(operator.output[0], 'mandatory'):
        return sum(1 for output_tensor in operator.output if output_tensor.mandatory)
    else:
        # Fallback: assume all outputs are mandatory
        return len(operator.output)


def get_plugin_resource_flags(operator, package_info):
    """
    Generate plugin resource configuration flags based on default settings.

    Args:
        operator: The operator object
        package_info: The package info object

    Returns:
        str: C++ resource flags value (without field name)
    """
    # Default to HVX resource for UseQHPI
    return "QHPI_RESOURCE_HVX"


def get_plugin_function_signature(operator):
    """
    Generate QHPI function signature declarations.

    Args:
        operator: The operator object

    Returns:
        str: C++ function signature declarations for QHPI
    """
    func_name = f"{operator.type_name.lower()}PluginExecute"

    return f"""// QHPI execute function declaration
static uint32_t {func_name}(
    QHPI_RuntimeHandle* runtime_handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs
);"""


def get_plugin_execute_signature(operator):
    """
    Generate QHPI execute function signature for implementation.

    Args:
        operator: The operator object

    Returns:
        str: C++ function signature for QHPI execute function
    """
    func_name = f"{operator.type_name.lower()}PluginExecute"

    return f"""static uint32_t {func_name}(
    QHPI_RuntimeHandle* runtime_handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs
)"""


def has_use_qhpi(operator, package_info):
    """
    Check if the operator should use QHPI based on UseQHPI flag.

    Args:
        operator: The operator object to check
        package_info: The package info object

    Returns:
        bool: True if the operator should use QHPI, False otherwise
    """
    # Check if package_info has use_qhpi flag set
    return hasattr(package_info, 'use_qhpi') and package_info.use_qhpi


def get_plugin_template_type():
    """
    Get the template type identifier for QHPI templates.

    Returns:
        str: Template type identifier
    """
    return "HTP_PLUGIN"


def generate_plugin_includes():
    """
    Generate the necessary include statements for QHPI code.

    Returns:
        str: C++ include statements for QHPI
    """
    return """// QHPI includes
#include "qhpi.h"
"""

def generate_plugin_cost_function(operator):
    """
    Generate a QHPI compatible cost function.

    Args:
        operator: The operator object

    Returns:
        str: C++ cost function implementation
    """
    op_name_lower = operator.type_name.lower()

    return f"""
__attribute__((unused)) static float {op_name_lower}CostFunc(const uint32_t num_inputs, const QHPI_Tensor* const* inputs)
{{
  /*
   * QHPI cost function implementation
   * This function estimates the computational cost of the operation
   * for the HTP scheduler to make optimal placement decisions.
   */

  float cost = 0.0;  // add cost computation here
  return cost;
}}
"""


def translate_datatype_to_qhpi_element_type(datatype):
    """
    Translate QNN DataType or QnnType to QHPI_Element_Type for QHPI templates.

    Based on the actual QHPI_Element_Type enum from hexnn_qhpi.h:
    QHPI_UNKNOWN = 0, QHPI_QUInt8 = 1, QHPI_QUInt16 = 2, QHPI_QInt16 = 3,
    QHPI_Float32 = 4, QHPI_Int32 = 5, QHPI_QInt32 = 6, QHPI_QInt8 = 7,
    QHPI_Float16 = 8, QHPI_Int64 = 9, QHPI_Any_Element_Type = 240, QHPI_PrepareError = 241

    Args:
        datatype: The QNN DataType object, string, or QNN_DATATYPE_* constant

    Returns:
        str: The corresponding QHPI_Element_Type constant
    """
    # Handle None or empty values
    if datatype is None or datatype == 'None' or datatype == '':
        return 'QHPI_UNKNOWN'

    # Handle different input types
    if hasattr(datatype, 'name'):
        datatype_name = datatype.name
    elif hasattr(datatype, 'type'):
        datatype_name = datatype.type
    else:
        datatype_name = str(datatype)

    # Handle None or empty string cases
    if not datatype_name or datatype_name == 'None':
        return 'QHPI_UNKNOWN'

    # Normalize QNN_DATATYPE_* constants to simple names
    if datatype_name.startswith('QNN_DATATYPE_'):
        datatype_name = datatype_name[13:]  # Remove 'QNN_DATATYPE_' prefix

    datatype_upper = datatype_name.upper()

    # Map QNN DataTypes to QHPI_Element_Type constants (exact match with hexnn_qhpi.h)
    datatype_mapping = {
        # Direct mappings (supported by QHPI)
        'FLOAT_32': 'QHPI_Float32',
        'FLOAT_16': 'QHPI_Float16',
        'INT_32': 'QHPI_Int32',
        'INT_64': 'QHPI_Int64',
        'UINT_8': 'QHPI_QUInt8',
        'UINT_16': 'QHPI_QUInt16',
        'UINT_32': 'QHPI_Int32',  # Map to closest available type
        'UINT_64': 'QHPI_Int64',  # Map to closest available type
        'INT_8': 'QHPI_QInt8',
        'INT_16': 'QHPI_QInt16',
        'QINT_8': 'QHPI_QInt8',
        'QINT_16': 'QHPI_QInt16',
        'QINT_32': 'QHPI_QInt32',
        'QUINT_8': 'QHPI_QUInt8',
        'QUINT_16': 'QHPI_QUInt16',
    }

    # Handle unsupported types with reasonable fallbacks
    unsupported_mappings = {
        # Fixed-point types -> map to closest integer type
        'UFIXED_POINT_8': 'QHPI_QUInt8',
        'UFIXED_POINT_16': 'QHPI_QUInt16',
        'UFIXED_POINT_32': 'QHPI_Int32',
        'SFIXED_POINT_8': 'QHPI_QInt8',
        'SFIXED_POINT_16': 'QHPI_QInt16',
        'SFIXED_POINT_32': 'QHPI_QInt32',

        # Legacy fixed-point naming
        'UFIXED_8': 'QHPI_QUInt8',
        'UFIXED_16': 'QHPI_QUInt16',
        'UFIXED_32': 'QHPI_Int32',
        'SFIXED_8': 'QHPI_QInt8',
        'SFIXED_16': 'QHPI_QInt16',
        'SFIXED_32': 'QHPI_QInt32',

        # Boolean -> map to UInt8
        'BOOL_8': 'QHPI_QUInt8',
        'BOOL': 'QHPI_QUInt8',
        'BOOLEAN': 'QHPI_QUInt8',

        # String -> not supported, use UNKNOWN
        'STRING': 'QHPI_UNKNOWN',

        # Undefined -> use UNKNOWN
        'UNDEFINED': 'QHPI_UNKNOWN',
    }

    # Check for direct mapping first
    if datatype_upper in datatype_mapping:
        return datatype_mapping[datatype_upper]

    # Check for unsupported type mappings
    if datatype_upper in unsupported_mappings:
        return unsupported_mappings[datatype_upper]

    # Check for generic patterns
    if datatype_upper.startswith('UFIXED'):
        return 'QHPI_Int32'
    elif datatype_upper.startswith('SFIXED'):
        return 'QHPI_QInt32'
    elif datatype_upper.startswith('BOOL'):
        return 'QHPI_QUInt8'

    # Default fallback
    return 'QHPI_Float32'


def get_signature_combinations(operator, package_info):
    """
    Generate default kernel combinations for UseQHPI operations.

    Args:
        operator: The operator object
        package_info: The package info object

    Returns:
        list: List of default combinations for UseQHPI operations
    """
    # For UseQHPI, return a simple default combination
    combinations = []

    # Extract data types from operator inputs/outputs
    input_types = []
    if operator.input:
        for input_tensor in operator.input:
            if hasattr(input_tensor, 'datatypes') and input_tensor.datatypes:
                input_types.extend(input_tensor.datatypes)

    output_types = []
    if operator.output:
        for output_tensor in operator.output:
            if hasattr(output_tensor, 'datatypes') and output_tensor.datatypes:
                output_types.extend(output_tensor.datatypes)

    # Generate a default combination
    if input_types or output_types:
        primary_type = input_types[0] if input_types else output_types[0]
        type_name = str(primary_type).replace('QnnDatatype.', '').replace('NativeDatatype.', '')
        suffix = f"_{type_name.lower()}_"

        combinations.append({
            'input_types': input_types,
            'output_types': output_types,
            'suffix': suffix,
            'description': f'Default UseQHPI combination with {type_name}',
        })

    return combinations


def should_use_plugin_template(operator, package_info):
    """
    Determine if the operator should use the QHPI template instead of DEF_* macros.

    This function checks if UseQHPI is enabled, which indicates that
    QHPI code should be generated instead of traditional DEF_* macro code.

    Args:
        operator: The operator object to check
        package_info: The package info object

    Returns:
        bool: True if QHPI template should be used, False for traditional DEF_* template
    """
    return has_use_qhpi(operator, package_info)


# Backward compatibility alias for the old function name
def translate_datatype_to_hexnn_element_type(datatype):
    """
    Backward compatibility alias for translate_datatype_to_qhpi_element_type.

    This function maintains backward compatibility for existing code that imports
    the old function name while redirecting to the new QHPI implementation.

    Args:
        datatype: The QNN DataType object, string, or QNN_DATATYPE_* constant

    Returns:
        str: The corresponding QHPI_Element_Type constant
    """
    return translate_datatype_to_qhpi_element_type(datatype)
