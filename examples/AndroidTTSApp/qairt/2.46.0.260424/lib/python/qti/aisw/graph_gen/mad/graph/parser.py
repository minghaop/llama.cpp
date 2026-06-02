# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
""" File contains the core logic of the DAG parser, core logic of the MAD library for generating structure """
import ast
import copy
import inspect
import operator
from pprint import pprint
import numpy


from qti.aisw.graph_gen.mad.graph.components import Graph, Tensor, Op, TensorTypes, ProducerTypes, TensorDataInfo, AllowedDtypes
from qti.aisw.graph_gen.mad.lib.op import QcSqrt, QcAdd, QcSub, QcDiv, QcMul


class DagParser:
    """
    Parses a Python model's 'forward' method to build a computational graph (DAG).

    This parser inspects the source code of a model's `forward` method,
    identifies operations (layers) and their connections, and constructs
    a `Graph` object representing the data flow. It handles variable assignments,
    functional calls, for loops, and conditional statements.

    :param layers[dict]: Dictionary of known layers/operations.
    :param submodules[dict]: Dictionary of submodules.
    :param attributes[dict]: Dictionary of model attributes.
    :param input_tensors[list]: List of input tensors.
    :param input_tensors_data_info[list]: List of data information for input tensors.
    :param graph[Graph]: The generated computational graph.
    :param var_to_tensor[dict]: Maps Python variable names to tensor names in the graph.
    :param var_groups[dict]: Stores groups of tensors for list-like variables.
    """
    def __init__(self, model, input_tensors=None, input_tensors_data_info=None, graph_name='model'):
        """
        Initializes the DagParser.

        :param model: The model object to parse.
        :param input_tensors[list, optional]: Pre-defined input tensors. Defaults to None.
        :param input_tensors_data_info[list, optional]: Data info for input tensors. Defaults to None.
        :param graph_name[str, optional]: The name of the graph. Defaults to 'model'.
        """
        # Store operations and submodules objects
        self.layers = model.layers
        self.submodules = model.submodules
        self.attributes = model.attributes

        if input_tensors is None:
            input_tensors = []
        self.input_tensors = input_tensors

        if input_tensors_data_info is None:
            input_tensors_data_info = []
        self.input_tensors_data_info = input_tensors_data_info

        # Store the graph structure
        self.graph = Graph(graph_name)

        # Variables names can be reused but each op will generate a new tensor.
        # Below map stores the current tensor name the variable is pointing to
        self.var_to_tensor = {}

        # Variable groups to hold multiple tensor together.
        # These are needed in case of cashing the instance of a particular varible inside a for loop so that it can be used while returning
        self.var_groups = {} # Variable : [List of tensors]


        # Make var for all the attributes having weight

    def get_op_name(self, prefix):
        """
        Helper method to create a unique name for an operator within the current graph scope.

        This is not the final operator name used in the graph, but a unique identifier
        during the parsing process.

        :param prefix[str]: The desired prefix for the operator name.
        :return: A unique operator name.
        """
        count = 0
        op_name = f'{prefix}_{count}'
        query = f'{self.graph.name}.{op_name}'
        while query in self.graph.ops:
            count += 1
            op_name = f'{prefix}_{count}'
            query = f'{self.graph.name}.{op_name}'
        return op_name

    def get_var_name(self, prefix):
        """
        Helper method to create a unique variable name.

        :param prefix[str]: The desired prefix for the variable name.
        :return: A unique variable name.
        """
        count = 0
        op_var = f'{prefix}_{count}'
        while op_var in self.var_to_tensor:
            count += 1
            op_var = f'{prefix}_{count}'
        return op_var

    def capture_dag(self, lvalue, operator, operands, type = "UNK", attrs=None, static_inputs=None):
        """
        Captures an operation and its connections to form part of the DAG.

        :param lvalue[list]: Left-hand side variables (outputs of the operation).
        :param operator[str]: The name of the operator.
        :param operands[list]: Right-hand side variables (inputs to the operation).
        :param type[str, optional]: The type of the operation. Defaults to "UNK".
        :param attrs[dict, optional]: Attributes of the operation. Defaults to None.
        :param static_inputs[list, optional]: List of static input tensor names. Defaults to None.
        """
        if static_inputs is None:
            static_inputs = []
        if attrs is None:
            attrs = {}
        # print('DAG: {} <- {} <- {}'.format(lvalue, operator, operands))

        # Process op only for leaf nodes. e.g. Conv, ReLU
        op_name = self.graph.name + "." + operator
        op = Op(op_name, type, attrs, static_inputs=static_inputs)

        # Process inputs : Add consumer in the input tensors
        for i, operand in enumerate(operands):
            # Fetch the tensor name from the variable if not raise error
            tensor_name = self.var_to_tensor.get(operand, None)
            assert tensor_name is not None, f"No tensor found corresponding to the variable {operand} :("

            # if operand not in self.var_to_tensor:
            #     self.var_to_tensor[operand] = tensor_name

            tensor = self.graph.tensors.get(tensor_name, None)
            if tensor is None:
                raise ValueError(f"Tensor: {tensor_name} corresponding to the variable {operand} not found in graph")

            tensor.consumer.append(op_name)
            if tensor_name not in self.graph.tensors:
                self.graph.tensors[tensor_name] = tensor

            # Add input info to the op
            op.inputs.append(tensor_name)


        # Process outputs: Add producer tensors
        for i, lval in enumerate(lvalue):
            tensor_name = op_name + "_output_" + str(i)

            # Tensor name of every op should be unique
            assert tensor_name not in self.graph.tensors , f"{tensor_name} already present in the graph"
            in_tensor_data_infos = []
            for tensor in op.inputs:
                in_tensor_data_infos.append(self.graph.tensors[tensor].data_info)

            out_tensor_info = self.layers[operator].output_tensor_info(in_tensor_data_infos)

            tensor = Tensor(tensor_name, TensorTypes.DYNAMIC, op_name, data_info=out_tensor_info)
            self.graph.tensors[tensor_name] = tensor

            # Update the variable tensor mapping
            self.var_to_tensor[lval] = tensor_name

            # Update Ops output tensor info
            op.outputs.append(tensor_name)

        # update op info in graph
        self.graph.ops[op_name] = op
        self.graph.ordered_ops.append(op.name)


    def parse_function(self, syn_tree):
        """
        Parses a function definition (expected to be the 'forward' method).

        This method extracts input arguments, associates them with input tensors,
        and then iterates through the function body to process assignments,
        returns, and for loops.

        :param syn_tree[ast.FunctionDef]: The AST node for the function definition.
        """

        # Only need to parse the forward function to get the graph structure.
        if syn_tree.name != 'forward':
            return

        # Forward Function parsing is divided into 3 parts.
        # 1. Parsing the input argument of the module which will act as the input tensors to the graph
        # 2. Parsing the rest of the operation statement to create the computation graph structure
        # 3. Parsing the return statement, this determines the output tensors of the graph

        input_arguments_list = []
        # First argument would always be self e.g.  "def forward(self, x):"
        # So except the first argument others defines actual tensor inputs of the graph
        for arg in syn_tree.args.args[1:]:
            input_arguments_list.append(arg.arg)

        if len(input_arguments_list) == 0:
            raise ValueError(" A graph should have at least one input, None found :( ")

        # Associate the input variables with the input tensors in case already provided.
        # Partial tensor list is not expected
        assert len(self.input_tensors)== 0 or len(self.input_tensors) == len(input_arguments_list), f"Incomplete Input tensor lists found for the graph {self.graph.name}"


        if len(self.input_tensors) == 0 and len(input_arguments_list) != len(self.input_tensors_data_info):
            raise ValueError(f"Tensor DataInfo should be provided for all the module input tensor, "
                             f"expected {len(input_arguments_list)} got {len(self.input_tensors_data_info)}")

        for i, input_argument in enumerate(input_arguments_list):

            if len(self.input_tensors) == 0:
                # In case the input is input tensors is not provided, create tensors for the input variables

                if isinstance(self.input_tensors_data_info[i], list):
                    # input variable could also be an input group, e.g, "def forward(self, input_ids, past_keys, past_values):"
                    # In the example if the input variable is a list of input then its corresponding TensorInfo would be a list.
                    # So in such cases populate the input tensors for each of the individual tensors as well as create the var_group for this group of tensors.
                    self.var_groups[input_argument] = []
                    for j, data_info in enumerate(self.input_tensors_data_info[i]):
                        tensor_name = f"{self.graph.name}.{input_argument}.{j}"
                        tensor = Tensor(tensor_name, TensorTypes.DYNAMIC, ProducerTypes.MODULE_INPUT, data_info=data_info)
                        self.var_groups[input_argument].append(tensor_name)
                        self._update_tensor_info(f'{input_argument}.{j}', tensor, tensor_name)

                elif isinstance(self.input_tensors_data_info[i], TensorDataInfo):
                    # For each individual tensor create Tensor object and update in tensor to variable info.
                    tensor_name = f"{self.graph.name}.{input_argument}"
                    tensor = Tensor(tensor_name, TensorTypes.DYNAMIC, ProducerTypes.MODULE_INPUT, data_info=self.input_tensors_data_info[i])
                    self._update_tensor_info(input_argument, tensor, tensor_name)

            else:
                # In case the input_tensor is already provided, update the tensor info
                tensor = self.input_tensors[i]
                tensor_name = self.input_tensors[i].name
                self._update_tensor_info(input_argument, tensor, tensor_name)

        # Process each statement of the function body as well as the return stateet
        for item in syn_tree.body:
            # Parse the assignment statement
            # e.g. out = self.linear(input)
            if isinstance(item, ast.Assign):

                # Assign statement can also be used for the variable assigment
                # e.g. skip = out
                # It can also be used to create a new variable group
                # e.g. output_keys = []
                if isinstance(item.value, (ast.List, ast.Name, ast.Tuple)):
                    self.process_variable_assignment(item)

                # Direct Simple Binary Operations like., +, -, * and / is not supported as of now.
                # e.g.  out = out * 0.5
                elif isinstance(item.value, ast.BinOp):
                    raise ValueError("Unsupported way of using binary operation detected. "
                                     "Please use ops from QcModules like QcAdd, QcSub, QcMul, QcDiv to perform operations")
                else:
                    self.parse_functional_assign(item)

            # Parse the Return Statement
            elif isinstance(item, ast.Return):
                self.parse_return(item)

                # Once the return statement is processed no need to parse more lines for this function.
                return

            # Special Call expression is supported named "ModuleMarker", these needs to be processed saperately
            # While defining the module make sure not to use any alias or calling from top level modeules
            # Examples of Invalid usage
            # =================================================================================================
            # from qti.aisw.graph_gen.mad.lib import module
            # ...
            # class Model(QcModule):
            # ...
            #   def forward(self, ...):
            #       ...
            #       module.ModuleMarker.split_at(out)  --> this will  error out
            #==================================================================================================
            #
            # from qti.aisw.graph_gen.mad.lib.module import ModuleMarker as marker
            # ...
            # class Model(QcModule):
            # ...
            #   def forward(self, ...):
            #       ...
            #       marker.split_at(out)  --> this will also error out
            elif isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
                if item.value.func.value.id == 'ModuleMarker':
                    self.process_module_marker(item)

            # To Parse For statement.  Basic For loop whose loop count is either constant or pre-defined by module attributes
            elif isinstance(item, ast.For):
                if isinstance(item.iter, ast.Call) and item.iter.func.id == 'range':
                    self.process_for_loop(item)
                else:
                    raise ValueError('Unsupported For loop found in forward pass :(')

            elif isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
                # This is the case for multiline comments, need to ignore.
                pass
            else:
                raise RuntimeError(f"Unsupported type of statement found:\n f{ast.unparse(item)}")

    def _update_tensor_info(self, variable_name, tensor, tensor_name):
        """
        Updates the internal mapping of variable names to tensor names and adds the tensor to the graph.

        :param variable_name[str]: The name of the Python variable.
        :param tensor[Tensor]: The Tensor object.
        :param tensor_name[str]: The unique name of the tensor in the graph.
        """
        self.var_to_tensor[variable_name] = tensor_name
        self.graph.tensors[tensor_name] = tensor
        self.graph.inputs.append(tensor_name)

    def process_variable_assignment(self, item: ast.Assign):
        """
        Processes an AST assignment node where the right-hand side is a variable, tuple, or list.

        This handles cases where variables are simply assigned the value of another variable
        or are part of a tuple/list unpacking.

        :param item: The AST assignment node.
        """

        if isinstance(item.value, ast.Name):
            # Variable assignment statement
            lvalue = [target.id for target in item.targets]
            assert len(lvalue) == 1

            rvalue = item.value.id
            # Assign the same tensor to the lvalue variable
            if rvalue in self.var_to_tensor:
                self.var_to_tensor[lvalue[0]] = self.var_to_tensor[rvalue]

        elif isinstance(item.value, ast.Tuple):
            # To parse statement like a, b = c, d
            lvalues = self._get_target_variables(item)

            rvalues = []
            for element in item.value.elts:
                rvalues.append(element.id)

            assert len(lvalues) == len(rvalues)

            for lvalue, rvalue in zip(lvalues, rvalues):
                if rvalue in self.var_to_tensor:
                    self.var_to_tensor[lvalue] = self.var_to_tensor[rvalue]

        elif isinstance(item.value, ast.List):
            # Creates a variable group
            # Statement example :
            #   outputs = []
            #   outputs = [past_key_1, past_value_1]

            lvalue = [target.id for target in item.targets]
            assert len(lvalue) == 1
            tenosors_list = []
            for element in item.value.elts:
                tenosors_list.append(self.var_to_tensor[element.id])

            self.var_groups[lvalue[0]] = tenosors_list

    def _get_target_variables(self, item: ast.Assign):
        """
        Extracts the names of target variables from an AST assignment item.

        :param item: The AST assignment node.
        :return: A list of variable names on the left-hand side of the assignment.
        """
        lvalues = []
        for target in item.targets:
            if isinstance(target, ast.Name):
                lvalues.append(target.id)
            elif isinstance(target, ast.Tuple):
                for element in target.elts:
                    lvalues.append(element.id)
        return lvalues

    def process_for_loop(self, item: ast.For):
        """
        Processes an AST for loop node.

        This method handles loops with `range` iterators, resolving the loop counter
        and then processing the body of the loop for each iteration. It also
        supports subscripted functional assignments, module markers, and variable
        group updates within the loop.

        :param item[ast.For]: The AST for loop node.
        :raises ValueError: If an unsupported type of for loop or attribute is found.
        :raises RuntimeError: If an unsupported type of statement is found within the loop.
        """
        loop_variable = item.target.id
        # Resolve the number of times the loop will run
        if isinstance(item.iter.args[0], ast.Constant):
            loop_counter =  item.iter.args[0].value
        elif isinstance(item.iter.args[0], ast.Attribute):
            if item.iter.args[0].value.id != 'self':
                raise ValueError("Unable to resolve loop counter, make sure only supported type of for loop is used in the forward definition ")
            loop_counter = self.attributes[item.iter.args[0].attr]

        for i in range(loop_counter):
            for loop_item in item.body:
                if isinstance(loop_item, ast.Assign) and isinstance(loop_item.value.func, ast.Subscript):
                    self.process_subscripted_functional_assign(loop_item, loop_variable, i)
                elif isinstance(loop_item, ast.Expr) and isinstance(loop_item.value, ast.Call):
                    if loop_item.value.func.value.id == 'ModuleMarker':
                        self.process_module_marker(loop_item)
                    else:
                        self.process_var_group_update(loop_item)
                elif isinstance(loop_item, ast.If) and isinstance(loop_item.test, ast.Compare):
                    if self._evaluate_test(loop_item.test, loop_variable, i):
                        statements_to_process =  loop_item.body
                    else:
                        statements_to_process =  loop_item.orelse

                    for statement in statements_to_process:
                        if isinstance(statement, ast.Assign) and isinstance(statement.value.func, ast.Subscript):
                            self.process_subscripted_functional_assign(statement, loop_variable, i)
                        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                            if statement.value.func.value.id == 'ModuleMarker':
                                self.process_module_marker(statement)
                            else:
                                self.process_var_group_update(statement)
                        elif isinstance(statement, ast.Break):
                            break
                        elif isinstance(statement, ast.Continue):
                            continue

    def process_module_marker(self, item):
        """
        Processes a ModuleMarker call, typically used to denote split points in the graph.

        :param item: The AST expression node representing the ModuleMarker call.
        :raises AssertionError: If an unsupported ModuleMarker attribute is found or if
                                the split marker has more than one variable.
        :raises ValueError: If the split marker has no variables.
        """
        assert item.value.func.attr == 'split_at', f"Unsupported ModuleMarker found : {item.value.func.attr}"

        variables = self._get_operands(item.value.args)
        if len(variables) == 0:
            ValueError("Split Marker should have at least one variable :(")
        split_group = []
        for variable in variables:
            split_group.append(self.var_to_tensor[variable])
        assert len(split_group) == 1, "For split only one tensor is supported"
        self.graph.split_info.append(split_group[0])



    def _evaluate_test(self, test, loop_variable, loop_counter):
        """
        Evaluates the condition of an 'if' statement within a loop.

        :param test[ast.Compare]: The AST comparison node.
        :param loop_variable[str]: The name of the loop variable.
        :param loop_counter[int]: The current value of the loop counter.
        :return: The result of the comparison.
        :raises AssertionError: If more than one comparison operator or comparator is found.
        :raises ValueError: If the left-hand side of the comparison is not a binary operation.
        """
        supported_comparision_op = {
            ast.Eq: operator.eq,
            ast.Gt: operator.gt,
            ast.Lt: operator.lt,
            ast.GtE: operator.ge,
            ast.LtE: operator.le,
            ast.NotEq: operator.ne,
        }

        assert len(test.ops) == 1, "Only one comparsion is supported in If statement test"
        operation = test.ops[0]
        assert type(operation) in supported_comparision_op, f"Unsupported comparison operation found {operation}"

        assert len(test.comparators) == 1, "Only one comparsion is supported in If statement test, found muliple comparators"
        comparator_value = self._get_value(test.comparators[0])

        if isinstance(test.left, ast.BinOp):
            eval_res = self._resolve_index_val_from_bin_op(test.left, loop_variable, loop_counter)
        elif isinstance(test.left, ast.Name):
            var = test.left.id
            assert var == loop_variable, "If Loop comparator can only use loop variables"
            eval_res = loop_counter
        else:
            raise ValueError("If condition left statement has to be a binary operation")

        return supported_comparision_op[type(operation)](eval_res, comparator_value)


    def _get_value(self, item):
        """
        Extracts the value from an AST node (Constant or Attribute).

        :param item[Union[ast.Constant, ast.Attribute]]: The AST node.
        :return: The extracted value.
        :raises ValueError: If an unsupported value type or an undefined attribute is found.
        """
        if isinstance(item, ast.Constant):
            return item.value
        elif isinstance(item, ast.Attribute):
            assert item.value.id == 'self', "Only class attribute is supported"
            attr = item.attr
            if attr not in self.attributes:
                raise ValueError(f"Undefined attribute '{attr}' found.")
            return self.attributes[attr]
        else:
            raise ValueError("Unsupported Value ")


    def process_var_group_update(self, loop_item):
        """
        Processes updates to variable groups within a loop, specifically handling 'append' operations.

        :param loop_item[ast.Expr]: The AST expression node representing the variable group update.
        :raises AssertionError: If an unsupported function attribute is called on the variable group.
        :raises RuntimeError: If the variable group is used before initialization.
        """
        assert loop_item.value.func.attr == 'append', f"Only append is supported on a variable group. Got {loop_item.value.func.attr}"

        var_group = loop_item.value.func.value.id
        update_variable = loop_item.value.args[0].id

        if var_group in self.var_groups:
            self.var_groups[var_group].append(self.var_to_tensor[update_variable])

            # Once we add a varible into a variable group .. only tensor info is added.
            # To access the variable for the subscripted access to an particular instance of a variable inside a
            # variable group an alias variable is created which is being used when accessed  a particular tensor.
            index = len(self.var_groups[var_group]) - 1
            alias_name = f'{var_group}.{index}'
            self.var_to_tensor[alias_name] = self.var_to_tensor[update_variable]

        else:
            raise RuntimeError(f"{var_group} used before initialization :(")

    def _resolve_index_val_from_bin_op(self, slice, loop_variable, loop_counter):
        """
        Resolves the index value from a binary operation within a subscript.

        :param slice[ast.BinOp]: The AST binary operation node.
        :param loop_variable[str]: The name of the loop variable.
        :param loop_counter[int]: The current value of the loop counter.
        :return: The resolved index value.
        :raises ValueError: If an unsupported operation for the subscript is found.
        """
        def _get_operands(operand):
            if isinstance(operand, ast.Name):
                var = operand.id
                assert var == loop_variable, f"Invalid loop variable found {var}, expected {loop_variable}"

                return loop_counter
            elif isinstance(operand, ast.Attribute):
                assert operand.value.id == 'self', "Unsupported attribute id found as a subscript"
                attribute = operand.attr
                if attribute not in self.attributes:
                    raise ValueError(f"Undefined attribute found {attribute}")
                return self.attributes[attribute]
            elif isinstance(operand, ast.Constant):
                return operand.value
            else:
                raise RuntimeError(f"Unsupported operand for binary op to resolve the index of subscript")

        supported_operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.FloorDiv: operator.floordiv,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod
        }

        left = _get_operands(slice.left) if not isinstance(slice.left, ast.BinOp) else self._resolve_index_val_from_bin_op(slice.left, loop_variable, loop_counter)
        right = _get_operands(slice.right) if not isinstance(slice.right, ast.BinOp) else self._resolve_index_val_from_bin_op(slice.right, loop_variable, loop_counter)
        op = slice.op

        if not isinstance(op, tuple(supported_operations.keys())):
            raise ValueError(f"Unsupported operation '{type(op)}' for the subscript.")

        op = supported_operations[type(op)]
        index = int(op(left, right))
        return index

    def _get_operands(self, args, loop_variable=None, loop_counter=None):
        """
        Extracts operands from AST arguments, handling names, attributes, subscripts, and starred expressions.

        :param args[list]: List of AST argument nodes.
        :param loop_variable[str, optional]: The name of the loop variable if inside a loop. Defaults to None.
        :param loop_counter[int, optional]: The current value of the loop counter if inside a loop. Defaults to None.
        :return: A list of operand names (strings).
        :raises RuntimeError: If a variable group is not found or an index is out of bounds.
        :raises ValueError: If an unknown argument type is encountered.
        """
        operands = []
        for arg in args:
            if isinstance(arg, ast.Name):
                operands.append(arg.id)
            elif isinstance(arg, ast.Attribute):
                operands.append(self._get_variable_for_attribute(arg))
            elif isinstance(arg, ast.Subscript):
                assert loop_variable is not None and loop_counter is not None
                if isinstance(arg.slice, ast.Name):
                    slice = arg.slice.id
                    assert slice == loop_variable
                    instance_index = loop_counter
                elif isinstance(arg.slice, ast.BinOp):
                    instance_index = self._resolve_index_val_from_bin_op(arg.slice, loop_variable, loop_counter)
                elif isinstance(arg.slice, ast.Constant) and isinstance(arg.slice.value, int):
                    instance_index = arg.slice.value
                else:
                    raise ValueError("Unsupported subscript variable type: ", type(arg.slice), ast.dump(arg.slice))

                var_group = arg.value.id
                if var_group not in self.var_groups:
                    raise RuntimeError(f"Variable group '{var_group}' not found")
                if len(self.var_groups[var_group]) <= instance_index:
                    raise RuntimeError(f"Index {instance_index} out of bounds for variable group '{var_group}' (size: {len(self.var_groups[var_group])})")
                operands.append(f'{var_group}.{instance_index}')

            elif isinstance(arg, ast.Starred):
                # Append all the variables for the tensor group
                var_group = arg.value.id
                if var_group not in self.var_groups or len(self.var_groups[var_group]) == 0:
                    raise RuntimeError("Invalid variable group usage found, used before declaration")
                # Fetch the variable from the var group.
                count = len(self.var_groups[var_group])
                for i in range(count):
                    variable_name = f'{var_group}.{i}'
                    operands.append(variable_name)
            else:
                raise ValueError("Only Name and Attribute are supported as an argument of the functional call :(")

        return operands
    def process_subscripted_functional_assign(self, item, loop_variable, loop_counter):
        """
        Processes an AST assignment node where the right-hand side is a functional call
        with a subscripted function (e.g., `self.layers[i](x)`).

        This handles cases where operations or submodules are called within a loop
        using an index.

        :param item[ast.Assign]: The AST assignment node.
        :param loop_variable[str]: The name of the loop variable.
        :param loop_counter[int]: The current value of the loop counter.
        :raises AssertionError: If the number of left-hand side variables does not match
                                the number of outputs from the submodule.
        """
        lvalue = self._get_target_variables(item)
        operator = item.value.func.value.attr
        if isinstance(item.value.func.slice, ast.Name):
            slice = item.value.func.slice.id
            assert slice == loop_variable
            operator_index = loop_counter
        elif isinstance(item.value.func.slice, ast.BinOp):
            operator_index = self._resolve_index_val_from_bin_op(item.value.func.slice, loop_variable, loop_counter)
        else:
            raise ValueError("Unsupported slice type :", type(item.value.func.slice))

        operator = f'{operator}.{operator_index}'
        operands = self._get_operands(item.value.args, loop_variable, loop_counter)

        if operator in self.submodules:
            submodule_name = f'{self.graph.name}.{operator}'
            operand_tensors = [self.graph.tensors[self.var_to_tensor[operand]] for operand in operands]
            output_graph = self.process_submodule(self.submodules[operator], operand_tensors, submodule_name = submodule_name)
            assert len(lvalue) == len(output_graph.outputs)

            # Update the variable to tensor map and add tensors in the current graph as well with
            for i, op_var in enumerate(lvalue):
                output_tensor_name = output_graph.outputs[i]
                self.var_to_tensor[op_var] = output_tensor_name
                self.graph.tensors[output_tensor_name] = copy.deepcopy(output_graph.tensors[output_tensor_name])
                self.graph.tensors[output_tensor_name].producer = submodule_name

            # Update the consumer to the input tensors
            for operand in operands:
                self.graph.tensors[self.var_to_tensor[operand]].consumer.append(submodule_name)

            # Add entry in the graph
            self.graph.ops[submodule_name] = output_graph
            self.graph.ordered_ops.append(submodule_name)

        else:
            type = 'UNK'
            attrs = {}
            if operator in self.layers:
                type = self.layers[operator].type
                attrs = self.layers[operator].get_attributes()

                # Add tensors for the static input for the op. e.g. for layers like Conv, Linear add the tensors for weight and bias.
                static_inputs = self.layers[operator].get_static_inputs()
                static_inputs_tensor_names = []
                if static_inputs:
                    for name, shape in static_inputs:
                        # Create Tensor object
                        # Note:: Variable is not created for the static inputs as they are only associated with a particular op
                        tensor_name = self.graph.name + "." + operator + '.' + name
                        input_tensor = Tensor(tensor_name, TensorTypes.STATIC, operator, data_info=shape)
                        input_tensor.consumer.append(operator)

                        # Add tensor object to the graph
                        self.graph.tensors[tensor_name] = input_tensor
                        static_inputs_tensor_names.append(tensor_name)

            self.capture_dag(lvalue, operator, operands, type, attrs, static_inputs_tensor_names)
        pass

    def parse_return(self, item):
        """
        Parses an AST return node and updates the graph's output tensors.

        :param item[ast.Return]: The AST return node.
        :raises ValueError: If an unknown return statement type is encountered.
        """
        # pprint(ast.dump(item.value))
        if isinstance(item.value, ast.Name):
            output_var = item.value.id
            self.graph.outputs.append(self.var_to_tensor[output_var])
        elif isinstance(item.value, ast.Tuple):
            for output_name in item.value.elts:
                if isinstance(output_name, ast.Name):
                    self.graph.outputs.append(self.var_to_tensor[output_name.id])
                elif isinstance(output_name, ast.Starred):
                    self.graph.outputs.extend(self.var_groups[output_name.value.id])

        else:
            raise ValueError("Unknown return statement type !!")

    def _get_variable_for_attribute(self, arg):
        """
        Retrieves or creates a tensor for a model attribute.

        :param arg[ast.Attribute]: The AST attribute node (e.g., `self.weight`).
        :return: The name of the tensor corresponding to the attribute.
        :raises ValueError: If an unknown model attribute is encountered.
        """
        if arg.attr not in self.attributes:
            raise ValueError('Unknown model attribute while parsing graph')
        var_name = self.graph.name + "." + arg.value.id + "." + arg.attr
        if var_name not in self.var_to_tensor:
            value =self.attributes[arg.attr]
            shape = None
            # TODO : Create a function to get the datatype of of the scalars and assign appropriate datatype for them
            dtype = AllowedDtypes.FLOAT32

            if isinstance(value, (int, float, bool)):
                shape = 1  # non list indicates scalar type constants.
            elif isinstance(value, list):
                val = numpy.array(value)
                value = val
                shape = val.shape
            elif isinstance(value, numpy.ndarray):
                shape = value.shape
                dtype = value.dtype

            tensor = Tensor(var_name, TensorTypes.STATIC, ProducerTypes.CONSTANT, value=value, data_info=TensorDataInfo(shape, dtype))

            self.var_to_tensor[var_name] = var_name

            self.graph.tensors[var_name] = tensor
        return var_name

    def parse_functional_assign(self, item):
        """
        Parses an AST assignment node where the right-hand side is a functional call
        (e.g., `output = self.layer(input)`).

        This method handles calls to both regular layers and submodules, recursively
        parsing submodules to integrate their graphs.

        :param item[ast.Assign]: The AST assignment node.
        :raises AssertionError: If the number of left-hand side variables does not match
                                the number of outputs from a submodule.
        """

        if item.value.func.value.id != 'self':
            raise ValueError("Only operations defined in the model initialization can be used. "
                             "Found ", ast.unparse(item.value.func))

        lvalue = self._get_target_variables(item)
        operator = item.value.func.attr
        operands = self._get_operands(item.value.args)

        if operator in self.submodules:
            submodule_name = f'{self.graph.name}.{operator}'
            operand_tensors = [self.graph.tensors[self.var_to_tensor[operand]] for operand in operands]
            output_graph = self.process_submodule(self.submodules[operator], operand_tensors, submodule_name = submodule_name)
            assert len(lvalue) == len(output_graph.outputs)

            # Update the variable to tensor map and add tensors in the current graph as well with
            for i, op_var in enumerate(lvalue):
                output_tensor_name = output_graph.outputs[i]
                self.var_to_tensor[op_var] = output_tensor_name
                self.graph.tensors[output_tensor_name] = copy.deepcopy(output_graph.tensors[output_tensor_name])
                self.graph.tensors[output_tensor_name].producer = submodule_name

            # Update the consumer to the input tensors
            for operand in operands:
                self.graph.tensors[self.var_to_tensor[operand]].consumer.append(submodule_name)

            # Add entry in the graph
            self.graph.ops[submodule_name] = output_graph
            self.graph.ordered_ops.append(submodule_name)

        else:
            type = 'UNK'
            attrs = {}
            if operator in self.layers:
                type = self.layers[operator].type
                attrs = self.layers[operator].get_attributes()

                # Add tensors for the static input for the op. e.g. for layers like Conv, Linear add the tensors for weight and bias.
                static_inputs = self.layers[operator].get_static_inputs()
                static_inputs_tensor_names = []
                if static_inputs:
                    for name, shape in static_inputs:
                        # Create Tensor object
                        # Note:: Variable is not created for the static inputs as they are only associated with a particular op
                        tensor_name = self.graph.name + "." + operator + '.' + name
                        input_tensor = Tensor(tensor_name, TensorTypes.STATIC, operator, data_info=shape)
                        input_tensor.consumer.append(operator)

                        # Add tensor object to the graph
                        self.graph.tensors[tensor_name] = input_tensor
                        static_inputs_tensor_names.append(tensor_name)

                self.capture_dag(lvalue, operator, operands, type, attrs, static_inputs_tensor_names)
            else:
                raise ValueError("Undefined Operation found. Please use QcModules ops only to define the graph")

    def parse_dag(self, syn_tree):
        """
        Parses the AST of the model to extract the computational graph.

        :param syn_tree[ast.Module]: The AST of the entire module (model class).
        """

        syn_tree = syn_tree.body[0]
        for item in syn_tree.body:
            if isinstance(item, ast.FunctionDef):
                self.parse_function(item)

    def process_submodule(self, model, input_tensors, submodule_name):
        """
        Recursively processes a submodule to parse its graph.

        :param model: The submodule model object.
        :param input_tensors[list]: List of input tensors for the submodule.
        :param submodule_name[str]: The name of the submodule.
        :return: The parsed graph of the submodule.
        """
        dag = DagParser(model, input_tensors,graph_name=submodule_name)
        output_graph = dag.parse_model(model)
        return output_graph


    def parse_ops(self, layers):
        """
        Recursively prints the types of operations in the layers dictionary.

        :param layers[dict]: A dictionary of layers (can be nested).
        """
        for layer, obj in layers.items():
            if not isinstance(obj, dict):
                print(f'{str(layer)} => {str(obj.type)}')
            else:
                self.parse_ops(obj)

    def parse_model(self, model):
        """
        Parses the entire model by inspecting its source code and building the DAG.

        :param model: The model object to parse.
        :return: The generated computational graph.
        """
        lines = inspect.getsource(model.__class__)
        ast_mod = ast.parse(lines)
        self.parse_dag(ast_mod)
        return self.graph
