# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the infrastructure of the post graph optimization,
and also the entry of the mha2sha optimization
"""

import copy
import hashlib
import json
import os
import pathlib
import traceback
import uuid
from collections import defaultdict, deque
from typing import Iterable

import onnx
import onnx_ir as ir
import yaml
from pydantic import BaseModel, ConfigDict, FilePath
from safetensors.numpy import load_file as load_safetensors_file
from safetensors.numpy import save_file as save_safetensors_file

from qairt.optimizer.onnx.utils.encodings import (
    GraphEncodingInfo,
    deserialize_encodings,
    save_encodings,
    serialize_graph_encodings,
)
from qairt.optimizer.onnx.utils.ir_extra_info import (
    GraphExtraInfo,
    VariableExtraInfo,
    chain_m2s_tracing_info,
    is_expert_encset_name,
    parse_expert_encset_name,
)
from qairt.optimizer.onnx.utils.utils import (
    copy_value,
    get_unique_name,
    hash_values,
    is_constant,
    iter_all_values,
    make_constant_node,
    move_external_constant_to_initializer,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class ExportedUseCase(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    name: str
    safetensors: FilePath
    encodings: FilePath | None = None


class ExportedFiles(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    onnx_path: FilePath
    data_path: FilePath
    encodings_path: FilePath | None = None
    use_cases: list[ExportedUseCase] = []
    lora_tensor_names: FilePath | None = None
    lora_importer_config: FilePath | None = None
    lora_transform_metadata_path: FilePath | None = None


class GraphContext:
    """
    Central data structure for ONNX graph optimization operations

    This class encapsulates an ONNX model along with its metadata like encodings,
    safetensors, and updatable tensors. It provides methods for embedding this
    metadata into the graph, extracting it, and serializing/deserializing the model

    All graph passes operate on this context object rather than directly on the graph
    """

    # pylint: disable=[R0913,R0917]
    def __init__(
        self,
        model_ir: ir.Model,  # external data is not required to load
        named_encodings: dict[str, dict] | None = None,
        named_safetensors: dict[str, dict] | None = None,
        updatable_tensors: list[str] | None = None,
        naming_prefix: str = "opt",
        _skip_shape_infer: bool = False,
    ):
        """Base class of the graph optimizer that focused on post-quantization optimization.
        Args:
            model: ir.Model instance
            named_encodings: a dictionary conatins all the encodings information of the graph
                            key is the enc_set name, value is the corresponding graph encodings
            named_safetensors: a dictionary conatins all the lora saftensors information of the graph
                            key is the enc_set name, value is the corresponding saftensors
            updatable_tensors: The updatable tensor names of the model.
            naming_prefix: The name prefix of the new node/tensor created by the optimizer.
        """

        # Use the provided ir.Model directly
        self.model_ir = model_ir
        self.graph_ir = self.model_ir.graph

        # Only initialize graph metadata if it doesn't exist (e.g., from splitting)
        if "extra_info" not in self.graph_ir.meta:
            self.graph_ir.meta["extra_info"] = GraphExtraInfo(naming_prefix=naming_prefix)
            self.graph_ir.meta["extra_info"].naming_policy.init_from_graph(self.graph_ir)

        # Initialize extra_info for all tensors if not already present (e.g., from splitting)
        # This ensures ShapeInference and other passes can safely access tensor metadata
        for v in iter_all_values(self.graph_ir):
            if "extra_info" not in v.meta:
                v.meta["extra_info"] = VariableExtraInfo()

        # Embed metadata if provided
        if named_safetensors:
            self.embed_safetensors_into_graph(named_safetensors)

        if updatable_tensors:
            self.embed_updatable_tensors_into_graph(updatable_tensors)

        # Run shape inference first on the graph as this will be used
        # to compute certain fields of encodings
        # Imported here to avoid circular import errors
        from qairt.optimizer.onnx.passes import ShapeInference

        if not _skip_shape_infer:
            ShapeInference().apply(self)

        if named_encodings:
            named_encodings = dict(named_encodings.items())
            src_named_encodings: dict[str, GraphEncodingInfo] = {}

            for enc_set_name, encodings in named_encodings.items():
                src_named_encodings[enc_set_name] = deserialize_encodings(encodings, self.model_ir)

            self.embed_encodings_into_graph(src_named_encodings)

        self.graph_ir.meta["extra_info"].validate()

    @classmethod
    def from_files(
        cls,
        model_path: str | os.PathLike,
        encodings_path: str | os.PathLike | None = None,
        lora_adapters_path: str | os.PathLike | None = None,
        lora_tensor_names_path: str | os.PathLike | None = None,
        naming_prefix: str = "opt",
        **kwargs,
    ):
        """Load the model and encodings from file and initialize a GraphContext.

        Args:
            model_path: Path to ONNX model.
            encodings_path: Path to AIMET encodings file. Supported versions are v0.6.1 and v1.0.0.
            lora_adapters_path: Path to LoRA adapters YAML file (lora_importer_config).

                YAML schema::

                    use_case:
                      - name: <usecase_1/adapter_1 name>
                        lora_weights: <path to safetensor file for adapter_1>
                        quant_overrides: <path to AIMET encodings file for adapter_1>
                      - name: <usecase_2/adapter_2 name>
                        lora_weights: <path to safetensor file for adapter_2>
                        quant_overrides: <path to AIMET encodings file for adapter_2>
                      ...

            lora_tensor_names_path: Path to .txt file with updatable LoRA tensor names.
        Keyword Args:
            log_level (str): Severity of logging level. Default is INFO.
        Returns:
            GraphContext: Initialized graph context.
        """

        model = ir.load(model_path)

        named_encodings = {}
        named_safetensors = {}
        updatable_tensors = []

        if lora_adapters_path:
            yaml_path = pathlib.Path(lora_adapters_path)
            yaml_dir = yaml_path.parent
            with open(yaml_path, "r") as f:
                lora_adapters_list = yaml.safe_load(f)["use_case"]

            for adapter in lora_adapters_list:
                adapter_name = adapter["name"]
                try:
                    # Resolve quant_overrides path relative to YAML directory
                    adapter_encodings_path = yaml_dir / adapter["quant_overrides"]
                    with open(adapter_encodings_path) as f:
                        encodings = json.load(f)
                    named_encodings[adapter_name] = encodings
                except KeyError:
                    pass

                # Resolve lora_weights path relative to YAML directory
                adapter_safetensors_path = yaml_dir / adapter["lora_weights"]
                named_safetensors[adapter_name] = load_safetensors_file(adapter_safetensors_path)

        if encodings_path:
            # Hypothetically, if "base" is the name of one of the adapters
            # This comes up with an alternate name for the base encodings
            base_enc_name = "base"
            while base_enc_name in named_encodings:
                base_enc_name += "_"

            with open(encodings_path, "r") as f:
                encodings = json.load(f)
            named_encodings[base_enc_name] = encodings

        if lora_tensor_names_path:
            with open(lora_tensor_names_path, "r") as f:
                updatable_tensors = [l.strip() for l in f.readlines()]

        return GraphContext(model, named_encodings, named_safetensors, updatable_tensors, naming_prefix)

    def embed_updatable_tensors_into_graph(self, updatable_tensors: list[str]):
        """Embed updatable tensors into ir graph

        Args:
            updatable_tensors: The updatable tensor names of the model.
        """
        updatable_tensors_set = set(updatable_tensors)

        for v in iter_all_values(self.graph_ir):
            if v.name in updatable_tensors_set:
                v.meta["extra_info"].is_updatable = True

    def embed_encodings_into_graph(self, named_encodings: dict[str, GraphEncodingInfo]):
        """Embed encodings into ir graph

        Args:
            named_encodings: a dictionary conatins all the encodings information of the graph
                             key is the enc_set name, value is the corresponding graph encodings
        """
        for encset_name, graph_enc in named_encodings.items():
            tensors_encodings = graph_enc.encodings

            for v in iter_all_values(self.graph_ir):
                if v.name in tensors_encodings:
                    v.meta["extra_info"].named_encodings[encset_name] = tensors_encodings[v.name]

            # Store encoding versions and quantizer_args in graph metadata for preservation across transformations
            # This is especially useful for splitting
            self.graph_ir.meta["extra_info"].encoding_versions[encset_name] = graph_enc.version
            self.graph_ir.meta["extra_info"].quantizer_args[encset_name] = graph_enc.quantizer_args

    def embed_safetensors_into_graph(self, named_safetensors: dict[str, dict]):
        """Embed safetensors into ir graph

        Args:
            named_safetensors: a dictionary conatins all the lora saftensors information of the graph
                               key is the enc_set name, value is the corresponding saftensors
        """
        for set_name, safetensors in named_safetensors.items():
            for v in iter_all_values(self.graph_ir):
                if v.name in safetensors:
                    v.meta["extra_info"].named_safetensors[set_name] = safetensors[v.name]

    def save_onnx(self, path: str, external_data: str | None = None):
        """saved the ir Graph into the onnx file.
        This function is much more memory efficient than onnx.save(get_onnx_proto(load_weights=True))

        Args:
            path: the path to save the onnx file
            external_data: the filename for external data. If None, defaults to basename[:-5] + ".data"
        """
        # bugfix for onnx_ir,
        # onnx_ir will not handle the path of external data for Constant node in the serialization,
        # move all constant to initializers (so that we don't need to load them)

        move_external_constant_to_initializer(self.model_ir)

        base_dir = os.path.dirname(path)
        os.makedirs(base_dir, exist_ok=True)

        if external_data is None:
            basename = os.path.basename(path)
            external_data = basename[:-5] + ".data"

        ir.save(self.model_ir, path, external_data=external_data)
        logger.debug("saved onnx to %s", path)

    def get_onnx_proto(self):
        """Serialize the model into onnx.ModelProto"""
        onnx_proto = ir.serde.serialize_model(self.model_ir)
        return onnx_proto

    def get_encodings(self):
        """Extract encodings from the extra_info of each tensor in the ir Graph"""

        named_graph_encodings: dict[str, GraphEncodingInfo] = {}
        for encset_name, version in self.graph_ir.meta["extra_info"].encoding_versions.items():
            # Create instance with the same version and quantizer_args as origin
            graph_encodings = GraphEncodingInfo(version=version)
            graph_encodings.quantizer_args = self.graph_ir.meta["extra_info"].quantizer_args.get(
                encset_name, {}
            )
            named_graph_encodings[encset_name] = graph_encodings

        for v in iter_all_values(self.graph_ir):
            if "extra_info" in v.meta:
                for encset_name, v_enc in v.meta["extra_info"].named_encodings.items():
                    # Only process encodings for encoding sets that were originally present
                    # This filters out redundant encodings that may have been copied to
                    # non-LoRA tensors (e.g., embedding Gather output when split_embedding=True)
                    if encset_name in named_graph_encodings:
                        named_graph_encodings[encset_name].add_tensor_encodings(v.name, v_enc)

                    elif is_expert_encset_name(encset_name):
                        # this encset is an internal expert encset for MoE model

                        # The generated expert encset name should follow the format:
                        #   $MoE${base_encset_name}${expert_id}
                        # For example: "$MoE$base$3"
                        #
                        # The "$" symbol is reserved (see RESERVED_ENCSET_SYMBOLS) and cannot be used by users
                        # when defining encset names.
                        #
                        # As a result, if the MoE pass is not enabled, `is_expert_encset_name` will always return False.

                        based_encset_name, expert_id = parse_expert_encset_name(encset_name)
                        named_graph_encodings[based_encset_name].add_MoE_expert_tensor_encodings(
                            v.name, expert_id, v_enc
                        )

        for graph_enc in named_graph_encodings.values():
            graph_enc.infer_MoE_dyn_expert_id_map(self.graph_ir)

        return named_graph_encodings

    def get_safetensors(self):
        """Extract saftensors from the extra_info of each tensor in the ir Graph"""
        named_safetensors: defaultdict[str, dict] = defaultdict(dict)
        # extract safetensors from extra_info
        for v in self.graph_ir.initializers.values():
            if "extra_info" in v.meta:
                for encset_name, safetensor in v.meta["extra_info"].named_safetensors.items():
                    named_safetensors[encset_name][v.name] = safetensor
        return named_safetensors

    def get_updatable_tensor_names(self):
        """Extract updatable tensors from the extra_info of each tensor in the ir Graph"""
        # extract updatable from extra_info
        updatable_tensor_names = []
        for v in iter_all_values(self.graph_ir):
            if "extra_info" in v.meta and v.meta["extra_info"].is_updatable:
                updatable_tensor_names.append(v.name)

        return updatable_tensor_names

    def save_tracing_info(self, path: str, merged=False):
        """
        Save tracing information to the file

        Args:
            path: the path to save the tracing information
            merged: whether to merge the chainable transformations into one
        """
        tracing_info_j = self.get_tracing_info(merged)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(tracing_info_j, f, indent=4)

    def get_tracing_info(self, merged=False):
        """
        Get tracing information of all transformations recorded

        Args:
            merged: whether to merge the chainable transformations into one
        """
        one2one_tracing_info = self.graph_ir.meta["extra_info"].one2one_tracing_info
        if merged:
            one2one_tracing_info = chain_m2s_tracing_info(one2one_tracing_info)

        tracing_info_j = []
        for _, v in one2one_tracing_info.items():
            tracing_info_j.append(v.as_dict())

        for v in self.graph_ir.meta["extra_info"].subgraph_tracing_info:
            tracing_info_j.append(v.as_dict())
        return tracing_info_j

    def export(
        self,
        path: str | os.PathLike,
        prefix: str = "model",
    ) -> ExportedFiles:
        """Export model artifacts
        Args:
            path: Directory where the artifacts are to be saved
            prefix: Prefix to model and artifact file names. Defaults to "model"
        """
        # 1. Export onnx_proto
        export_path = pathlib.Path(path)

        if not export_path.exists():
            os.makedirs(export_path, exist_ok=True)

        elif not export_path.is_dir():
            raise OSError(f"{export_path} is not a directory")

        model_path = export_path / f"{prefix}.onnx"
        data_file = f"{prefix}.data"
        data_location = export_path / data_file

        self.save_onnx(str(model_path), external_data=data_file)
        logger.info(f"Model saved at {model_path.absolute()}")

        # Instance of ExportedFiles to store the path to exported files
        exported_files: ExportedFiles = ExportedFiles(onnx_path=model_path, data_path=data_location)

        try:
            onnx.checker.check_model(model_path, full_check=True)
            logger.debug("ONNX Checker passed!")
        except Exception:
            logger.warning(f"ONNX checker failed!")
            traceback.print_exc()
            logger.warning("Please re-check the model before using")

        # 2 (Optional) Export safetensors
        named_safetensors = self.get_safetensors()

        use_cases = {}
        exported_use_cases = {}

        for set_name, safetensors in named_safetensors.items():
            safetensors_path = export_path / f"{set_name}.safetensors"
            save_safetensors_file(safetensors, safetensors_path)

            logger.debug(f"Safetensors of adapter '{set_name}' saved at {safetensors_path.absolute()}")

            use_case = {
                "name": set_name,
                "model_name": str(model_path.absolute()),
                "output_path": "../lora_output",
                "lora_weights": str(safetensors_path.absolute()),
            }
            use_cases[set_name] = use_case
            exported_use_cases[set_name] = ExportedUseCase(name=set_name, safetensors=safetensors_path)

        # 3(Optional) Export encodings
        named_encodings = self.get_encodings()

        if named_encodings:
            # Determine base encoding name (encoding not in named_safetensors)
            base_enc_name = None
            for encset_name in named_encodings:
                if encset_name not in named_safetensors:
                    base_enc_name = encset_name
                    break

            # Export all encodings
            for encset_name, enc in named_encodings.items():
                if encset_name == base_enc_name:  # Base encodings
                    encodings_path = export_path / f"{prefix}.encodings"
                    save_encodings(enc, encodings_path)
                    logger.info(f"Encodings saved at {encodings_path.absolute()}")
                    exported_files.encodings_path = encodings_path
                elif encset_name in use_cases:  # Adapter encodings (only if corresponding safetensor exists)
                    encodings_path = export_path / f"{encset_name}.encodings"
                    use_cases[encset_name]["quant_overrides"] = str(encodings_path.absolute())
                    save_encodings(enc, encodings_path)
                    logger.debug(f"Encodings of adapter '{encset_name}' saved at {encodings_path.absolute()}")
                    exported_use_cases[encset_name].encodings = encodings_path

        if use_cases:
            lora_importer_config: dict[str, list] = {"use_case": []}

            for _, config in use_cases.items():
                lora_importer_config["use_case"].append(config)

            lora_importer_config_path = export_path / "lora_importer_config.yaml"
            with open(lora_importer_config_path, "w") as f:
                f.write(yaml.dump(lora_importer_config))

            exported_files.use_cases = [use_case for _, use_case in exported_use_cases.items()]
            exported_files.lora_importer_config = lora_importer_config_path

        # 4 (Optional) Updatable tensors list
        updatable_tensors = self.get_updatable_tensor_names()

        if updatable_tensors:
            lora_tensor_names_path = export_path / "lora_tensor_names.txt"
            with open(lora_tensor_names_path, "w") as f:
                for tensor in updatable_tensors:
                    f.write(f"{tensor}\n")

            exported_files.lora_tensor_names = lora_tensor_names_path

        return exported_files


def get_subgraph_full_func_name(func_name: str, hash_id: str):
    return func_name + "__{}".format(hash_id[:16])


def parse_subgraph_full_func_name(full_func_name: str) -> tuple[str, str]:
    splits = full_func_name.split("__")
    return "__".join(splits[:-1]), splits[-1]


class SubGraphDesc:
    class SubGraphDescError(Exception):
        pass

    class NotValidSubGraphDesc(SubGraphDescError):
        pass

    class SubGraphUnbounded(SubGraphDescError):
        pass

    def __init__(
        self,
        graph: ir.Graph,
        inputs: list[ir.Value],
        outputs: list[ir.Value],
        nodes: list[ir.Node],
        allow_external_user: bool = True,
        meta: dict | None = None,
    ):
        """
        allow_external_user: whether allow internal tensors to be used externally
                             note: some internal tensors maybe constant actually and shared by many users (internal or external)
        meta: any additional information that we want to embed into the subgraph, note: this information will not used by this class,
              and will not propagated to the graph created by this class
        """
        self.graph: ir.Graph = graph
        self.inputs: list[ir.Value] = inputs
        self.outputs: list[ir.Value] = outputs
        self.nodes: list[ir.Node] = nodes  # non topo sorted nodes list are acceptable
        self.initializers: dict[str, ir.Value] = {}  # will be infered automatically in validate()
        if meta is None:
            meta = {}
        self.meta = meta
        self.validate(allow_external_user)

    @classmethod
    def infer_by_io(
        cls,
        graph: ir.Graph,
        inputs: list[ir.Value],
        outputs: list[ir.Value],
        max_layers_to_traverse: int | None = 100,
        allow_external_user: bool = True,
    ):
        return SubGraphDesc(
            graph=graph,
            inputs=inputs,
            outputs=outputs,
            nodes=SubGraphDesc.get_bounded_nodes(
                inputs=inputs, outputs=outputs, max_layers_to_traverse=max_layers_to_traverse
            ),
            allow_external_user=allow_external_user,
        )

    @classmethod
    def get_bounded_nodes(
        cls, inputs: list[ir.Value], outputs: list[ir.Value], max_layers_to_traverse: int | None = 100
    ):
        reversed_nodes: list[ir.Node] = []
        node_id_set: set[int] = set()
        # run a BFS layer scan bottom-up,
        # if the scan is not finished within layer_budget, then raise a "SubGraphUnbounded" error
        # the typical reason is the subgraph cannot be bounded by the given inputs/outputs

        visited: set[int] = set()
        queue: deque[tuple[ir.Value | None, int]] = deque(
            [(v, 0) for v in outputs]
        )  # (node, current_layer_id)
        inputs_id_set: set[int] = set(id(x) for x in inputs)
        while queue:
            current_v, current_layer_id = queue.popleft()

            if current_v is None:
                continue
            if id(current_v) in visited:
                continue
            if id(current_v) in inputs_id_set:
                continue
            visited.add(id(current_v))

            if max_layers_to_traverse and current_layer_id >= max_layers_to_traverse:
                raise cls.SubGraphUnbounded()

            producer = current_v.producer()
            if producer is None:
                continue
            if id(producer) in node_id_set:
                continue
            reversed_nodes.append(producer)
            node_id_set.add(id(producer))
            for input_value in producer.inputs:
                if id(input_value) not in visited:
                    queue.append((input_value, current_layer_id + 1))

        nodes = list(reversed(reversed_nodes))

        # nodes may not topo-sorted
        return nodes

    def validate(self, allow_external_user=True):
        """
        Validate the subgraph:
        - no intermediate tensor can be used externally
        - tensors used internally is not produced externally
        - inputs cannot be output
        - no duplicate nodes
        nodes can be not toposorted
        """
        internal_node_id_set = set(id(x) for x in self.nodes)
        inputs_id_set = set(id(x) for x in self.inputs)
        outputs_id_set = set(id(x) for x in self.outputs)
        internal_tensors_id_set = set()

        for n in self.nodes:
            for v in n.outputs:
                internal_tensors_id_set.add(id(v))

        for n in self.nodes:
            for v in n.inputs:  # type: ignore
                if v is None:
                    continue
                if id(v) in inputs_id_set:
                    continue
                if v.is_initializer():
                    if v.name not in self.initializers:
                        assert v.name is not None  # check for mypy
                        self.initializers[v.name] = v
                    continue
                if id(v) in internal_tensors_id_set:
                    continue
                raise self.SubGraphUnbounded(
                    f"SubGraph is not bounded,the internal node '{n.name}' uses external tensor '{v.name}'"
                )

            if not allow_external_user:
                for v in n.outputs:
                    if id(v) in outputs_id_set:
                        continue
                    for use in v.uses():
                        if id(use.node) not in internal_node_id_set:
                            raise self.SubGraphUnbounded(
                                "SubGraph is not bounded,"
                                f"the internal tensor '{v.name}' is used externally by '{use.node.name}'"
                            )

        for v in self.inputs:
            if id(v) in outputs_id_set:
                raise self.NotValidSubGraphDesc(
                    f"SubGraph input '{v.name}' cannot be output in the same time"
                )

        n_id_set = set()
        for n in self.nodes:
            if id(n) in n_id_set:
                raise self.NotValidSubGraphDesc(f"node '{n.name}' is duplicated in the SubGraph")
            n_id_set.add(id(n))

    def copy_as_graph(self, new_graph_name, replace_initializer_by_constant=False) -> ir.Graph:
        new_graph, ctx = GraphCopyCtx.copy_as_new_graph(
            graph_desc=self,
            new_graph_name=new_graph_name,
            replace_initializer_by_constant=replace_initializer_by_constant,
            run_topo_sort=True,  # nodes in SubgraphDesc are not forcely sorted
        )
        return new_graph

    def replace_as_function(
        self,
        model: ir.Model,
        func_name,
        domain="qairt_tools.opt.custom",
        opset_version=1,
        attributes: Iterable[ir.Attr] | None = None,
        with_full_name=True,
    ):
        # graph in function can't have initializers (will be ignored)
        # so we need to replace intializer by Constant ops

        hash_str = hash_values(self.outputs)[-16:]
        subgraph_name = get_subgraph_full_func_name(func_name, hash_str)
        overload = hash_str
        subgraph = self.copy_as_graph(subgraph_name + ".graph", replace_initializer_by_constant=True)
        if attributes is None:
            attributes = []

        if with_full_name:
            op_type_name = subgraph_name
        else:
            op_type_name = func_name

        ir_func = ir.Function(
            domain=domain, name=op_type_name, overload=overload, graph=subgraph, attributes=attributes
        )
        model.functions[ir_func.identifier()] = ir_func

        self.graph.opset_imports[domain] = max(self.graph.opset_imports.get(domain, 1), opset_version)

        # create func call node
        outputs = [copy_value(self.graph, v, v.name) for v in self.outputs]
        call_node = ir.Node(
            domain=domain,
            op_type=op_type_name,
            overload=overload,
            inputs=self.inputs,
            outputs=outputs,
            attributes=attributes,
        )
        self.graph.insert_after(self.nodes[-1], call_node)
        for v, new_v in zip(self.outputs, outputs):
            safe_replace_all_uses_with(self.graph, v, new_v)

        return call_node


class GraphCopyCtx:
    def __init__(
        self,
        src_graph_desc: SubGraphDesc,
        dst_graph: ir.Graph,
        insert_after_dst_node: ir.Node | None,
        replace_initializer_by_constant=False,
        run_topo_sort=True,
    ):
        self.replace_initializer_by_constant = replace_initializer_by_constant
        self.run_topo_sort = run_topo_sort
        self.src_graph_desc = src_graph_desc
        self.dst_graph = dst_graph
        self.insert_after_dst_node = insert_after_dst_node
        self.v_map: dict[int, ir.Value] = {}

    @classmethod
    def copy_as_new_graph(
        cls,
        graph_desc: SubGraphDesc,
        new_graph_name: str,
        replace_initializer_by_constant=False,
        run_topo_sort=True,
    ):
        dst_graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            name=new_graph_name,
            opset_imports=graph_desc.graph.opset_imports.copy(),
        )
        dst_graph.meta["extra_info"] = graph_desc.graph.meta["extra_info"].copy_for_subgraph("")
        ctx = cls.copy_into(
            graph_desc,
            dst_graph,
            None,
            replace_initializer_by_constant=replace_initializer_by_constant,
            run_topo_sort=run_topo_sort,
        )

        # handle inputs/outputs
        for v in ctx.src_graph_desc.inputs:
            copied_v = ctx.v_map[id(v)]
            dst_graph.inputs.append(copied_v)
        for v in ctx.src_graph_desc.outputs:
            copied_v = ctx.v_map[id(v)]
            dst_graph.outputs.append(copied_v)

        return dst_graph, ctx

    @classmethod
    def copy_into(
        self,
        src_graph_desc: SubGraphDesc,
        dst_graph: ir.Graph,
        insert_after_dst_node: ir.Node | None,
        replace_initializer_by_constant=False,
        run_topo_sort=True,
    ) -> "GraphCopyCtx":
        ctx = GraphCopyCtx(
            src_graph_desc,
            dst_graph,
            insert_after_dst_node=insert_after_dst_node,
            replace_initializer_by_constant=replace_initializer_by_constant,
            run_topo_sort=run_topo_sort,
        )
        ctx._copy_subgraph()
        return ctx

    def _copy_subgraph(self):
        """
        Copy Subgraph describted by graph_desc into dst_graph
        if insert_after_node is None, then new nodes will be inserted at the top

        note: the inputs/outpus of the subgraph are not added as dst_graph's inputs/outputs
        """

        new_nodes = []
        for v in self.src_graph_desc.inputs:
            new_v = self._copy_value(v)

        for v in self.src_graph_desc.initializers.values():
            if not self.replace_initializer_by_constant:
                new_v = self._copy_value(v)
                self.dst_graph.initializers.add(new_v)
            else:
                assert v.const_value is not None  # check for mypy
                assert v.name is not None
                cst_node = ir.Node(
                    "",
                    "Constant",
                    inputs=[],
                    outputs=[self._copy_value(v)],
                    attributes=[ir.AttrTensor("value", v.const_value)],
                    name=get_unique_name(self.dst_graph, v.name + "/Constant"),
                )
                new_nodes.append(cst_node)

        # nodes in self.src_graph_desc maybe not topo sorted,
        # so we firstly copy all values
        for n in self.src_graph_desc.nodes:
            for v in n.outputs:
                self._copy_value(v)

        for n in self.src_graph_desc.nodes:
            new_n = self._copy_node(n)
            new_nodes.append(new_n)

        # insert new nodes into self.dst_graph
        if self.insert_after_dst_node is None:
            if len(self.dst_graph) == 0:
                self.dst_graph.extend(new_nodes)
            else:
                self.dst_graph.insert_before(self.dst_graph[0], new_nodes)
        else:
            self.dst_graph.insert_after(self.insert_after_dst_node, new_nodes)

        if self.run_topo_sort:
            self.dst_graph.sort()  # make it topo-sorted

    def _copy_value(self, v: ir.Value):
        new_v = ir.Value(
            producer=None,
            name=get_unique_name(self.dst_graph, v.name),
            shape=copy.deepcopy(v.shape),
            type=copy.deepcopy(v.type),
            doc_string=copy.deepcopy(v.doc_string),
            const_value=self._copy_constant_value(v.const_value),
            metadata_props=self._copy(v.metadata_props),
        )
        new_v.meta.update(self._copy(v.meta))
        self.v_map[id(v)] = new_v
        return new_v

    def _copy_node(self, n: ir.Node):
        new_inputs = [self.v_map[id(x)] if x is not None else None for x in n.inputs]
        new_n = ir.Node(
            domain=n.domain,
            op_type=n.op_type,
            inputs=new_inputs,
            attributes=[self._copy_attr(x) for x in n.attributes.values()],
            overload=n.overload,
            outputs=[self.v_map[id(x)] for x in n.outputs],
            version=n.version,
            name=get_unique_name(self.dst_graph, n.name),
            doc_string=n.doc_string,
            metadata_props=self._copy(n.metadata_props),
        )
        new_n.meta.update(self._copy(n.meta))
        return new_n

    def _copy_constant_value(self, item: ir.TensorProtocol | None):
        # to save memory and speedup, we don't copy const_value,
        # note: obj of TensorProtocol is actually immutable
        return item

    def _copy_attr(self, item: ir.Attr):
        return ir.Attr(
            name=item.name,
            type=item.type,
            value=self._copy(item.value),
            ref_attr_name=item.ref_attr_name,
            doc_string=item.doc_string,
        )

    def _copy(self, item):
        # isinstance and deepcopy attr relatively expensive operation
        # try to call directly _copy_attr, _copy_constant_value if possible
        if isinstance(item, ir.Attr):
            return self._copy_attr(item)
        if isinstance(item, ir.TensorProtocol):
            return self._copy_constant_value(item)
        return copy.deepcopy(item)


def inline_function(model_ir: ir.Model, callers: ir.Node | list[ir.Node]):
    """
    inline one or multiple call nodes for same function

    the constants/initializers in the function will be shared in the main graph
    """
    if isinstance(callers, ir.Node):
        callers = [callers]

    main_graph = callers[0].graph
    assert main_graph is not None  # check for mypy

    # those callers should share same function
    for op in callers[1:]:
        assert op.op_identifier() == callers[0].op_identifier()

    subgraph = model_ir.functions[callers[0].op_identifier()].graph

    # copy subgraph into main graph
    shared_cst_in_main_graph: dict[
        int, ir.Value
    ] = {}  # key is the id of ir.Value in the subgraph, value is the ir.Value in main graph
    for caller_n in callers:
        copy_ctx = GraphCopyCtx.copy_into(
            SubGraphDesc(
                graph=subgraph,
                inputs=list(subgraph.inputs),
                outputs=list(subgraph.outputs),
                nodes=list(subgraph),
            ),
            main_graph,
            insert_after_dst_node=caller_n,
            run_topo_sort=False,
        )

        for subgraph_input_v, caller_input_v in zip(subgraph.inputs, caller_n.inputs):
            copied_v = copy_ctx.v_map[id(subgraph_input_v)]
            safe_replace_all_uses_with(main_graph, copied_v, caller_input_v)

        for subgraph_output_v, caller_output_v in zip(subgraph.outputs, caller_n.outputs):
            copied_v = copy_ctx.v_map[id(subgraph_output_v)]
            safe_replace_all_uses_with(main_graph, caller_output_v, copied_v)

        if len(shared_cst_in_main_graph) == 0:
            for v in iter_all_values(subgraph):
                if is_constant(v):
                    shared_cst_in_main_graph[id(v)] = copy_ctx.v_map[id(v)]
        else:
            for id_v_in_subgraph, shared_v_in_maingraph in shared_cst_in_main_graph.items():
                safe_replace_all_uses_with(
                    main_graph, copy_ctx.v_map[id_v_in_subgraph], shared_v_in_maingraph
                )

    return copy_ctx
