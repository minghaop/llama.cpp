<%doc>
//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
// =============================================================================
</%doc>

<%page expression_filter="n" expression_filter="trim" />
<%!
from qti.aisw.op_package_generator.helpers.template_helpers import get_hexnn_tensor_sig, get_hexnn_param_sig,_template_builder, is_valid_cpp_identifier, get_param_order_sig, build_scalar_param
import itertools

def get_plugin_config_from_package(package_info):
    """Get plugin config from package_info since it's stored at package level"""
    return getattr(package_info, 'plugin_config', None)

# Import the updated helper functions from plugin_helpers
from qti.aisw.op_package_generator.helpers.plugin_helpers import get_plugin_config_flags, get_plugin_resource_flags, translate_datatype_to_hexnn_element_type, get_signature_combinations

def translate_layout_to_qhpi(layout):
    """Translate PluginSignature layout to QHPI_Standard_Layout constant"""
    if not layout:
        return "QHPI_Layout_Flat4"

    layout_str = str(layout).upper()
    # Map common layout patterns to QHPI_Standard_Layout constants (from hexnn_qhpi.h)
    if "NCHW" in layout_str:
        return "QHPI_Layout_NCHW"
    elif "NHWC" in layout_str or "FLAT4" in layout_str:
        return "QHPI_Layout_Flat4"  # NHWC is typically represented as Flat4 in QHPI
    elif "FLAT5" in layout_str:
        return "QHPI_Layout_Flat5"
    elif "FLAT6" in layout_str:
        return "QHPI_Layout_Flat6"
    elif "CROUTON" in layout_str:
        if "8" in layout_str:
            return "QHPI_Layout_Crouton_8"
        elif "16" in layout_str:
            return "QHPI_Layout_Crouton_16"
        elif "32" in layout_str:
            return "QHPI_Layout_Crouton_32"
        else:
            return "QHPI_Layout_Crouton_8"  # Default crouton
    elif "ANY" in layout_str:
        return "QHPI_Layout_Any"
    else:
        return "QHPI_Layout_Flat4"  # Default fallback

def translate_storage_to_qhpi(storage):
    """Translate PluginSignature storage to QHPI_Storage constant"""
    if not storage:
        return "QHPI_Storage_Direct"

    storage_str = str(storage).upper()
    if "DIRECT" in storage_str and "INDIRECT" in storage_str:
        return "QHPI_Storage_Direct_OR_Indirect"
    elif "DIRECT" in storage_str:
        return "QHPI_Storage_Direct"
    elif "INDIRECT" in storage_str:
        return "QHPI_Storage_Indirect"
    else:
        return "QHPI_Storage_Direct"  # Default fallback

def translate_mem_placement_to_qhpi(mem_placement):
    """Translate PluginSignature mem_placement to QHPI_MemLoc constant"""
    if not mem_placement:
        return "QHPI_MemLoc_DDR_OR_TCM"

    mem_str = str(mem_placement).upper()
    if "DDR" in mem_str and "TCM" in mem_str:
        return "QHPI_MemLoc_DDR_OR_TCM"
    elif "DDR" in mem_str:
        return "QHPI_MemLoc_DDR_Only"
    elif "TCM" in mem_str:
        return "QHPI_MemLoc_TCM_Only"
    else:
        return "QHPI_MemLoc_DDR_OR_TCM"  # Default fallback

def get_signature_fields_from_combination(combination, index, is_input=True):
    """Extract signature fields from combination if available"""
    if 'kernel_config' not in combination:
        return None, None, None

    kernel_config = combination['kernel_config']
    signatures = kernel_config.input_signature if is_input else kernel_config.output_signature

    if not signatures or index >= len(signatures):
        return None, None, None

    signature = signatures[index]
    layout = getattr(signature, 'layout', None)
    storage = getattr(signature, 'storage', None)
    mem_placement = getattr(signature, 'mem_placement', None)

    return layout, storage, mem_placement

def generate_datatype_combinations(operator, package_info):
    """Generate combinations based on plugin signatures when available, otherwise use default logic"""
    # First, try to get signature-based combinations
    signature_combinations = get_signature_combinations(operator, package_info)
    if signature_combinations:
        return signature_combinations

    # Fall back to original logic if no signatures are available
    return generate_default_datatype_combinations(operator)

def generate_default_datatype_combinations(operator):
    """Generate combinations where input and output data types are exactly the same"""
    # Get allowed data types for each input tensor
    input_allowed_types_per_tensor = []
    if operator.input:
        for input_tensor in operator.input:
            if hasattr(input_tensor, 'allowed_data_types') and input_tensor.allowed_data_types:
                input_allowed_types_per_tensor.append(set(input_tensor.allowed_data_types))
            elif hasattr(input_tensor, 'data_type') and input_tensor.data_type:
                input_allowed_types_per_tensor.append(set([input_tensor.data_type]))
            else:
                # Default fallback
                input_allowed_types_per_tensor.append(set(['FLOAT_32']))

    # Get allowed data types for each output tensor
    output_allowed_types_per_tensor = []
    if operator.output:
        for output_tensor in operator.output:
            if hasattr(output_tensor, 'allowed_data_types') and output_tensor.allowed_data_types:
                output_allowed_types_per_tensor.append(set(output_tensor.allowed_data_types))
            elif hasattr(output_tensor, 'data_type') and output_tensor.data_type:
                output_allowed_types_per_tensor.append(set([output_tensor.data_type]))
            else:
                # Default fallback
                output_allowed_types_per_tensor.append(set(['FLOAT_32']))

    combinations = []

    # If no inputs or outputs, provide a default combination
    if not input_allowed_types_per_tensor and not output_allowed_types_per_tensor:
        combinations.append({
            'input_types': [],
            'output_types': [],
            'suffix': '_default_',
            'description': 'Default kernel (no inputs/outputs)'
        })
        return combinations

    # Find common data types that are supported by ALL tensors (inputs and outputs only)
    all_tensor_type_sets = input_allowed_types_per_tensor + output_allowed_types_per_tensor

    if all_tensor_type_sets:
        # Find intersection of all allowed data types - only types supported by ALL tensors
        common_data_types = set.intersection(*all_tensor_type_sets)

        # If no common types found, fall back to individual tensor type sets
        if not common_data_types:
            # Use union of all types and generate combinations where each tensor uses compatible types
            all_possible_types = set.union(*all_tensor_type_sets)
            common_data_types = all_possible_types

        # Generate one combination for each common data type
        # Sort by string representation since QnnDatatype objects can't be compared directly
        for data_type in sorted(common_data_types, key=lambda x: str(x)):
            # Create input types list - all inputs use the same data type
            input_types = [data_type] * len(input_allowed_types_per_tensor)
            # Create output types list - all outputs use the same data type
            output_types = [data_type] * len(output_allowed_types_per_tensor)

            # Generate a suffix based on the data type
            if hasattr(data_type, 'name'):
                type_name = data_type.name
            else:
                type_name = str(data_type)

            # Remove QNN_DATATYPE_ prefix if present
            if type_name.startswith('QNN_DATATYPE_'):
                type_name = type_name[13:]

            suffix = f"_{type_name.lower()}_"

            combinations.append({
                'input_types': input_types,
                'output_types': output_types,
                'suffix': suffix,
                'description': f'Kernel for {data_type} data type (all tensors same type)'
            })

    return combinations

def get_kernel_name_suffix(combination):
    """Generate a suffix for kernel names based on data type combination"""
    return combination.get('suffix', '')

def get_min_inputs(operator):
    """Calculate minimum number of mandatory input tensors including parameters"""
    input_count = 0

    # Count regular input tensors
    if operator.input:
        if hasattr(operator.input[0], 'mandatory'):
            input_count += sum(1 for input_tensor in operator.input if input_tensor.mandatory)
        else:
            # Fallback: assume all inputs are mandatory
            input_count += len(operator.input)

    # Count parameters as additional inputs
    if operator.param:
        input_count += len(operator.param)

    return input_count

def get_min_outputs(operator):
    """Calculate minimum number of mandatory output tensors"""
    if not operator.output:
        return 0
    # Check if TensorInfo has mandatory attribute, otherwise use all outputs as mandatory
    if hasattr(operator.output[0], 'mandatory'):
        return sum(1 for output_tensor in operator.output if output_tensor.mandatory)
    else:
        # Fallback: assume all outputs are mandatory
        return len(operator.output)

def get_param_data_type(param):
    """Extract parameter data type with fallback logic"""
    if hasattr(param, 'data_type') and param.data_type:
        return param.data_type
    elif hasattr(param, 'allowed_data_types') and param.allowed_data_types:
        return param.allowed_data_types[0]
    else:
        # Default based on parameter name patterns
        param_name_lower = param.name.lower() if hasattr(param, 'name') else ''
        if 'int' in param_name_lower or 'count' in param_name_lower or 'size' in param_name_lower:
            return 'UINT_32'
        elif 'float' in param_name_lower or 'scale' in param_name_lower or 'factor' in param_name_lower:
            return 'FLOAT_32'
        else:
            return 'UINT_32'

%>
<%
is_valid_cpp_identifier(operator.type_name.lower())
# Generate all data type combinations for this operator
datatype_combinations = generate_datatype_combinations(operator, package_info)
%>
//==============================================================================
// Auto Generated Code for ${operator.type_name} - QHPI Implementation
// Multiple kernels generated for different data type combinations
//==============================================================================

#include "HTP/core/constraints.h"
#include <string>

// Plugin/QHPI includes - using correct header from hexnn_qhpi.h
#include "HTP/core/qhpi.h"

%for combination in datatype_combinations:
<%
kernel_suffix = get_kernel_name_suffix(combination)
kernel_name = operator.type_name.lower() + kernel_suffix
%>
// Forward declarations for ${operator.type_name} kernel ${kernel_name}
static uint32_t ${kernel_name}Execute(QHPI_RuntimeHandle *handle,
                                      uint32_t num_outputs, QHPI_Tensor **outputs,
                                      uint32_t num_inputs, const QHPI_Tensor *const *inputs);
static float ${kernel_name}CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs);

%endfor
// Common forward declarations for ${operator.type_name}
static const QHPI_Op* ${operator.type_name.lower()}EarlyRewrite(const QHPI_Op *op);
static QHPI_Shape ${operator.type_name.lower()}ShapeRequired(const QHPI_Op *op);
static QHPI_Shape ${operator.type_name.lower()}ShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape);
static const QHPI_Op* ${operator.type_name.lower()}BuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent);
static const QHPI_Op* ${operator.type_name.lower()}LateRewrite(const QHPI_Op *op);

/*
 * QHPI Registration using hexnn_ffi.h API for ${operator.type_name}
 * Multiple kernels for different data type combinations
 */

%for combination in datatype_combinations:
<%
kernel_suffix = get_kernel_name_suffix(combination)
kernel_name = operator.type_name.lower() + kernel_suffix
input_types = combination['input_types']
output_types = combination['output_types']
%>
// Input tensor signatures for ${operator.type_name} kernel ${kernel_name}
// Includes both regular inputs and parameters as inputs
static QHPI_Tensor_Signature_v1 ${kernel_name}InputSignatures[] = {
%for i, input_type in enumerate(input_types):
<%
# Get signature fields from plugin config if available
layout, storage, mem_placement = get_signature_fields_from_combination(combination, i, is_input=True)
%>
    {
        .element_type = ${translate_datatype_to_hexnn_element_type(input_type)},
        .layout = ${translate_layout_to_qhpi(layout)},
        .storage = ${translate_storage_to_qhpi(storage)},
        .mem_placement = ${translate_mem_placement_to_qhpi(mem_placement)}
    }${"," if i < len(input_types) - 1 else ""}
%endfor
%if operator.param:
%for param in operator.param:
<%
# Extract parameter data type with multiple fallback options
param_data_type = get_param_data_type(param)
%>
    ,{
        .element_type = ${translate_datatype_to_hexnn_element_type(param_data_type)},
        .layout = QHPI_Layout_Any,
        .storage = QHPI_Storage_Direct_OR_Indirect,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
%endfor
%endif
};

static QHPI_Tensor_Signature_v1 ${kernel_name}OutputSignatures[] = {
%for i, output_type in enumerate(output_types):
<%
# Get signature fields from plugin config if available
layout, storage, mem_placement = get_signature_fields_from_combination(combination, i, is_input=False)
%>
    {
        .element_type = ${translate_datatype_to_hexnn_element_type(output_type)},
        .layout = ${translate_layout_to_qhpi(layout)},
        .storage = ${translate_storage_to_qhpi(storage)},
        .mem_placement = ${translate_mem_placement_to_qhpi(mem_placement)}
    }${"," if i < len(output_types) - 1 else ""}
%endfor
};

// Kernel definition for ${operator.type_name} kernel ${kernel_name}
static QHPI_Kernel_v1 ${kernel_name}Kernel = {
    .function_name = "${kernel_name}Execute",
    .function = ${kernel_name}Execute,
    .resources = ${get_plugin_resource_flags(operator, package_info)},
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = ${get_min_inputs(operator)},
    .input_signature = ${kernel_name}InputSignatures,
    .min_outputs = ${get_min_outputs(operator)},
    .output_signature = ${kernel_name}OutputSignatures,
    .cost_function = ${kernel_name}CostFunc,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate = nullptr
};

%endfor
// Array of all kernels for ${operator.type_name}
static QHPI_Kernel_v1 ${operator.type_name.lower()}Kernels[] = {
%for i, combination in enumerate(datatype_combinations):
<%
kernel_suffix = get_kernel_name_suffix(combination)
kernel_name = operator.type_name.lower() + kernel_suffix
%>
    ${kernel_name}Kernel${"," if i < len(datatype_combinations) - 1 else ""}
%endfor
};

// Operator info for ${operator.type_name} - exported for package registration
QHPI_OpInfo_v1 ${operator.type_name.lower()}OpInfo = {
    .name = THIS_PKG_NAME_STR "::" "${operator.type_name}",
    .num_kernels = ${len(datatype_combinations)},
    .kernels = ${operator.type_name.lower()}Kernels,
    .early_rewrite = ${operator.type_name.lower()}EarlyRewrite,
    .shape_required = ${operator.type_name.lower()}ShapeRequired,
    .shape_legalized = ${operator.type_name.lower()}ShapeLegal,
    .build_tile = ${operator.type_name.lower()}BuildTile,
    .late_rewrite = ${operator.type_name.lower()}LateRewrite
};

%for combination in datatype_combinations:
<%
kernel_suffix = get_kernel_name_suffix(combination)
kernel_name = operator.type_name.lower() + kernel_suffix
input_types = combination['input_types']
output_types = combination['output_types']
%>
/* QHPI execute function implementation for ${operator.type_name} kernel ${kernel_name} */
static uint32_t ${kernel_name}Execute(QHPI_RuntimeHandle *handle,
                                      uint32_t num_outputs, QHPI_Tensor **outputs,
                                      uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
  /*
   * QHPI implementation code for ${operator.type_name} kernel ${kernel_name}
   * This kernel handles the following data type combination:
%for i, input_type in enumerate(input_types):
   *   Input ${i}: ${input_type if input_type else 'UNKNOWN'} -> ${translate_datatype_to_hexnn_element_type(input_type) if input_type else 'QHPI_UNKNOWN'}
%endfor
%if operator.param:
%for i, param in enumerate(operator.param):
<%
# Extract parameter data type with multiple fallback options (same logic as above)
param_data_type = get_param_data_type(param)
%>
   *   Parameter ${i} (as input ${len(input_types) + i}): ${param.name} -> ${translate_datatype_to_hexnn_element_type(param_data_type)}
%endfor
%endif
%for i, output_type in enumerate(output_types):
   *   Output ${i}: ${output_type if output_type else 'UNKNOWN'} -> ${translate_datatype_to_hexnn_element_type(output_type) if output_type else 'QHPI_UNKNOWN'}
%endfor
   *
   * Input parameters:
   * - handle: Runtime handle for accessing runtime context
   * - num_outputs: Number of output tensors
   * - outputs: Array of output tensor pointers
   * - num_inputs: Number of input tensors (includes regular inputs + parameters)
   * - inputs: Array of input tensor pointers (regular inputs first, then parameters)
   */

  // Add your QHPI implementation for ${operator.type_name} kernel ${kernel_name} here
  // This kernel should handle the specific data type combination listed above

  return QHPI_Success;
}

static float ${kernel_name}CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
  /*
   * Cost estimation function for ${operator.type_name} kernel ${kernel_name}
   * Return approximate number of cycles needed for this operation
   * with the specific data type combination. Used for estimating cycle
   * performance of a graph.
   *
   * Parameters:
   * - num_inputs: Number of input tensors
   * - inputs: Array of input tensor pointers
   */

  float cost = 1000.0;  // add cost computation here based on tensor sizes and data types
  return cost;
}

%endfor
/*
 * Common stub implementations for ${operator.type_name} QHPI_OpInfo functions
 * These are shared across all kernels and provide default no-op implementations
 */

static const QHPI_Op* ${operator.type_name.lower()}EarlyRewrite(const QHPI_Op *op)
{
  /*
   * Early rewrite function for ${operator.type_name}
   * Called during graph optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

static QHPI_Shape ${operator.type_name.lower()}ShapeRequired(const QHPI_Op *op)
{
  /*
   * Shape required function for ${operator.type_name}
   * Specifies required input shapes for the operation
   * Return empty shape if no specific shape requirements
   */
  QHPI_Shape empty_shape = {0};  // Empty shape by default
  return empty_shape;
}

static QHPI_Shape ${operator.type_name.lower()}ShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape)
{
  /*
   * Shape legal function for ${operator.type_name}
   * Validates if a given shape is legal for this operation
   * Return the shape if legal, or modified shape if not legal
   */
  return *shape;  // Accept the provided shape by default
}

static const QHPI_Op* ${operator.type_name.lower()}BuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent)
{
  /*
   * Build tile function for ${operator.type_name}
   * Creates a tiled version of the operation
   * Return a new op that operates on the specified tile, or op if tiling is not supported
   */
  return op;  // No tiling support by default
}

static const QHPI_Op* ${operator.type_name.lower()}LateRewrite(const QHPI_Op *op)
{
  /*
   * Late rewrite function for ${operator.type_name}
   * Called during late optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

// Array of all ${operator.type_name} operations for registration
static QHPI_OpInfo_v1 ${operator.type_name.lower()}_ops[] = {
    ${operator.type_name.lower()}OpInfo
};

// Registration function for ${operator.type_name} operations
extern "C" void register_${operator.type_name.lower()}_ops()
{
    qhpi_register_ops_v1(sizeof(${operator.type_name.lower()}_ops) / sizeof(${operator.type_name.lower()}_ops[0]), ${operator.type_name.lower()}_ops, THIS_PKG_NAME_STR);
}
