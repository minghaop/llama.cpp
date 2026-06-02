# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Definitions of ir extra info and helper functions related to extra_info
"""

import copy
from enum import Enum

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.utils.encodings import TensorEncodingInfo


class AxisDenotation(str, Enum):
    """
    Axis denotation values for tensor dimensions

    Denotation describes what each dimension represents in the context of LLM models.
    This aligns with ONNX's dimension denotation concept, specialized for LLMs.

    Values:
        BATCH: Batch size dimension
        SEQ_LENGTH: Current sequence length (autoregressive length)
        PAST_SEQ_LENGTH: Past sequence length in KV cache
        CONTEXT_LENGTH: Global context length (PAST_SEQ_LENGTH + SEQ_LENGTH)
        SLIDING_CONTEXT_LENGTH: Sliding-window context length for SWA models
        UNKNOWN: Dimension meaning is unknown or not yet inferred
    """

    BATCH = "BATCH"
    SEQ_LENGTH = "SEQ_LENGTH"
    PAST_SEQ_LENGTH = "PAST_SEQ_LENGTH"
    CONTEXT_LENGTH = "CONTEXT_LENGTH"
    SLIDING_CONTEXT_LENGTH = "SLIDING_CONTEXT_LENGTH"
    UNKNOWN = "UNKNOWN"


RESERVED_ENCSET_SYMBOLS = ["$"]  # these symbols are reserved internally, cannot be used in the encset name


def parse_expert_encset_name(expert_encset_name: str):
    splits = expert_encset_name.split("$")
    if len(splits) != 4:
        return None, 0
    if splits[0] != "" or splits[1] != "MoE":
        return None, 0
    try:
        expert_id = int(splits[3])
    except Exception:
        return None, 0
    return splits[2], expert_id


def is_expert_encset_name(expert_encset_name: str) -> bool:
    based_encset_name, expert_id = parse_expert_encset_name(expert_encset_name)
    return based_encset_name is not None


def get_expert_encset_name(based_encset_name: str, expert_id: int):
    # note $ is a preserved symbol of encset name (check RESERVED_ENCSET_SYMBOLS)
    # so this name will not conflict with the name that given by user
    return f"$MoE${based_encset_name}${expert_id}"


class BaseTracingInfo:
    """
    Basic class to represent tracing information
    """

    def __init__(self):
        self.passes = []  # pass name

        # the relation ship on data between src and out
        self.same_data_after_proc = False
        # the relation ship on encodings between src and out
        self.same_encodings_after_proc = False

    def as_dict(self):
        """
        Serialized to dictionary
        """

        info_j = {"tracing_type": type(self).__name__}
        for k, v in self.__dict__.items():
            info_j[k] = v

        return info_j


class BaseOne2OneTracingInfo(BaseTracingInfo):
    """
    Basic class to represent ONE-TO-ONE tracing information
    """

    def __init__(self, src_name, dst_name, same_data_after_proc=False, same_encodings_after_proc=False):
        super().__init__()
        # represents One Variable to One Variable transformation
        assert isinstance(src_name, str)
        assert isinstance(dst_name, str)

        self.src_name = src_name
        self.dst_name = dst_name

        # the relation ship on data between src and out
        self.same_data_after_proc = same_data_after_proc
        # the relation ship on encodings between src and out
        self.same_encodings_after_proc = same_encodings_after_proc

        # note: in most of the time, same_date means same_encodings
        #       but in some special case, for example in AdaptiveMask,
        #       using 'dummy reshape' to indicate a re-quantize op,
        #       same data but not same encodings

    @classmethod
    def slicing(cls, src_name, dst_name, pass_name, axis, start, end, batch_slice_id, head_slice_id):
        # pylint: disable=[too-many-arguments,too-many-positional-arguments]
        """
        Record slicing as the tracing info
        """
        info = M2sTracingInfo(src_name, dst_name)
        info.axes.append(axis)
        info.starts.append(start)
        info.ends.append(end)
        info.batch_slice_ids.append(batch_slice_id)
        info.head_slice_ids.append(head_slice_id)
        info.passes.append(pass_name)
        return info

    @classmethod
    def sharing_encodings(cls, src_name, dst_name, pass_name, same_data_after_proc=False):
        """
        Record dst and src share same encodings
        """
        info = ShareEncodingsTracingInfo(src_name, dst_name)
        info.passes.append(pass_name)
        info.same_data_after_proc = same_data_after_proc
        return info


class M2sTracingInfo(BaseOne2OneTracingInfo):
    """
    Class to represents MHA2SHA tracing information
    """

    def __init__(self, src_name, dst_name):
        super().__init__(src_name, dst_name)
        # for MHA2SHA, dst_name should have same data/encodings of slice(src_name)
        self.same_data_after_proc = True
        self.same_encodings_after_proc = True

        self.axes = []
        self.starts = []
        self.ends = []
        self.batch_slice_ids = []
        self.head_slice_ids = []


class ShareEncodingsTracingInfo(BaseOne2OneTracingInfo):
    """
    Class to represents sharing encodings tracing information, typically used in layout transformation
    """

    def __init__(self, src_name, dst_name):
        super().__init__(src_name, dst_name)
        # M2sTracingInfo can be used to represent slicing/copy
        # however, in some case, the src/out are not equal, but sharing same encodings information
        # so record it by EncodingsTracingInfo

        self.same_data_after_proc = False
        self.same_encodings_after_proc = True


# pylint: disable=[too-many-instance-attributes]


class SubgraphTracingInfo(BaseTracingInfo):
    """
    Class to represents tracing info for complex transformation,
    for example, a src_subgraph is transformed into a dst_subgraph
    """

    # pylint: disable=[too-many-positional-arguments, too-many-arguments]

    def __init__(
        self,
        pass_name,
        src_subgraph_inputs,
        src_subgraph_outputs,
        src_intermediate_vars,
        dst_subgraph_inputs,
        dst_subgraph_outputs,
        dst_intermediate_vars,
    ):
        super().__init__()
        self.same_data_after_proc = False
        self.same_encodings_after_proc = False

        self.passes.append(pass_name)
        self.src_subgraph_inputs = src_subgraph_inputs
        self.src_subgraph_outputs = src_subgraph_outputs
        # attetion, not all intermediate vars will be removed from the graph
        #           may be used by others
        self.src_intermediate_vars = src_intermediate_vars
        self.dst_subgraph_inputs = dst_subgraph_inputs
        self.dst_subgraph_outputs = dst_subgraph_outputs
        self.dst_intermediate_vars = dst_intermediate_vars


class GraphExtraInfo:
    """
    Graph extra information
    """

    def __init__(self, naming_prefix):
        self.naming_policy = NamingPolicy(prefix=naming_prefix)
        # key is the dst_name of tracing info
        self.one2one_tracing_info: dict[str, BaseOne2OneTracingInfo] = {}
        self.subgraph_tracing_info: list[SubgraphTracingInfo] = []

        # Encoding metadata that needs to be preserved across splits
        self.encoding_versions: dict[str, str] = {}  # encset_name -> version
        self.quantizer_args: dict[str, dict] = {}  # encset_name -> quantizer_args

    def validate(self):
        for encset_name in self.encoding_versions.items():
            for symbol in RESERVED_ENCSET_SYMBOLS:
                assert symbol not in encset_name, (
                    f"symbol '{symbol}' is reserved internally, please change a name for encset {encset_name}"
                )

    def copy_for_subgraph(self, naming_prefix):
        new_extra_info = GraphExtraInfo(naming_prefix=naming_prefix)
        new_extra_info.encoding_versions = self.encoding_versions
        new_extra_info.quantizer_args = self.quantizer_args
        return new_extra_info

    def get_unique_name(self, namehint):
        """
        Allocate a new unique name of the graph
        """
        return self.naming_policy.get_unique_name(namehint)

    def get_unique_name_with_suffix(self, namehint: str | None, suffix: str):
        """
        Allocate a new unique name of the graph
        """
        if namehint is None:
            return self.naming_policy.get_unique_name("tmp" + suffix)

        return self.naming_policy.get_unique_name(namehint + suffix)

    # pylint: disable=[too-many-positional-arguments, too-many-arguments]
    def record_subgraph_transform(
        self,
        pass_name,
        src_subgraph_inputs,
        src_subgraph_outputs,
        src_intermediate_vars,
        dst_subgraph_inputs,
        dst_subgraph_outputs,
        dst_intermediate_vars,
    ):
        """
        Record complex subgraph transformation
        """
        info = SubgraphTracingInfo(
            pass_name,
            src_subgraph_inputs,
            src_subgraph_outputs,
            src_intermediate_vars,
            dst_subgraph_inputs,
            dst_subgraph_outputs,
            dst_intermediate_vars,
        )
        self.subgraph_tracing_info.append(info)

    # pylint: disable=[too-many-positional-arguments, too-many-arguments]
    def record_slicing(
        self,
        src_name,
        dst_name,
        pass_name,
        axis=None,
        start=None,
        end=None,
        batch_slice_id=None,
        head_slice_id=None,
    ):
        """
        Record the generated dst is a slice of src
        """
        # tracing
        info = BaseOne2OneTracingInfo.slicing(
            src_name,
            dst_name,
            pass_name=pass_name,
            axis=axis,
            start=start,
            end=end,
            batch_slice_id=batch_slice_id,
            head_slice_id=head_slice_id,
        )
        self.one2one_tracing_info[dst_name] = info

    def record_sharing_encodings(self, src_name, dst_name, pass_name):
        """
        Record the generated dst shares same encodings as src
        """
        # the recorded data relationship is UNKNOWN
        info = BaseOne2OneTracingInfo.sharing_encodings(src_name, dst_name, pass_name)
        self.one2one_tracing_info[dst_name] = info

    def record_copy(self, src_name, dst_name, pass_name):
        """
        Record the generated dst should have exactly same numerical value/encodings/safetensors as src
        """
        return self.record_slicing(src_name, dst_name, pass_name)


def chain_m2s_tracing_info(tracing_info: dict[str, M2sTracingInfo]):
    """
    Chain mha2sha tracing information
    So that the intermediate tensors will be eliminated in the tracing info
    """

    # merged_tracing_info = {}
    merged_tracing_info = dict(tracing_info.items())
    # while len(non_merged_tracing_info) > 0:
    #     for dst_name, info in non_merged_tracing_info.items():

    for dst_name, info in merged_tracing_info.items():
        src_name = info.src_name
        if (
            src_name in merged_tracing_info
            and isinstance(info, M2sTracingInfo)
            and isinstance(merged_tracing_info[src_name], M2sTracingInfo)
        ):
            prev_info = merged_tracing_info[src_name]
            tmp_info = copy.deepcopy(info)
            tmp_info.src_name = prev_info.src_name
            tmp_info.axes = prev_info.axes + info.axes
            tmp_info.starts = prev_info.starts + info.starts
            tmp_info.ends = prev_info.ends + info.ends
            tmp_info.batch_slice_ids = prev_info.batch_slice_ids
            tmp_info.head_slice_ids = prev_info.head_slice_ids
            tmp_info.passes = prev_info.passes + info.passes
            tmp_info.same_data_after_proc &= prev_info.same_data_after_proc
            tmp_info.same_encodings_after_proc &= prev_info.same_encodings_after_proc
            merged_tracing_info[dst_name] = tmp_info
        else:
            merged_tracing_info[dst_name] = info
    return merged_tracing_info


class SplittableParam:
    def __init__(self, axis: int, start: int, end: int):
        self.axis = axis
        self.start = start
        self.end = end

    def __eq__(self, other) -> bool:
        if not isinstance(other, SplittableParam):
            return False
        return self.axis == other.axis and self.start == other.start and self.end == other.end


class VariableExtraInfo:
    """
    Extra information for each Variable
    """

    def __init__(self):
        self.named_encodings: dict[str, TensorEncodingInfo] = {}
        self.named_safetensors: dict[str, np.ndarray] = {}
        self.is_updatable = False
        self._infered_constant_value: np.ndarray | None = None
        self.splittable: list[SplittableParam] = []
        self.axis_denotations: list[AxisDenotation] = []

    @property
    def infered_constant_value(self):
        """
        The constant value infered by this framework, typically used in shape inference
        """
        # making self._infered_constant_value immutable
        if self._infered_constant_value is not None:
            return self._infered_constant_value.copy()

        return None

    @infered_constant_value.setter
    def infered_constant_value(self, v: np.ndarray):
        self._infered_constant_value = v.copy()

    def merge(self, other, encodings_only=False):
        """
        Merge current extra info with other extra info
        if encodings_only is True, then only merge encodings

        This function is used typically for the copy case (encodings_only=False)
        or share encodings case (encodings_only=True)

        Args:
            other (VariableExtraInfo): the other extra info to merge
            encodings_only (bool, optional): whether to merge encodings only. Defaults to False.
        """
        if other is None:
            return
        self.named_encodings.update(other.named_encodings)
        if not encodings_only:
            self.named_safetensors.update(other.named_safetensors)
            self.is_updatable = self.is_updatable or other.is_updatable

    def defined_encodings(self):
        """
        Check whether encodings are defined for this tensor
        """
        # if saftensors is defined, then encodings must be defined
        return len(self.named_encodings) > 0

    def __eq__(self, other):
        if self.named_safetensors != other.named_safetensors:
            return False
        if self.named_encodings != other.named_encodings:
            return False
        if self.is_updatable != other.is_updatable:
            return False
        return True

    def is_updatable_weight(self):
        """
        Check whether this tensor has updatable weight
        """
        return len(self.named_safetensors) > 0

    def copy(self, ignore_safetensors=False):
        """
        Make a copy of the extra info

        Args:
            ignore_safetensors (bool, optional): whether to ignore safetensors. Defaults to False.
        """
        new_extra_info = copy.deepcopy(self)
        new_extra_info._infered_constant_value = None
        if ignore_safetensors:
            new_extra_info.named_safetensors = {}
        return new_extra_info

    def slice(self, axis=None, start=None, end=None):
        """
        Slice the extra info, just as slice the tensor
        the encodings and safetensors will be automatically sliced

        Args:
            axis: slice axis
            start: slice start
            end: slice end
        Returns:
            VariableExtraInfo
        """
        sliced_extra_info = copy.deepcopy(self)
        sliced_extra_info._infered_constant_value = None
        sliced_extra_info.named_encodings = {}
        sliced_extra_info.named_safetensors = {}

        # slice encodings - delegate to encoding's slice method
        for encset_name, enc in self.named_encodings.items():
            # Each encoding handles its own slicing logic based on enc_type
            sliced_extra_info.named_encodings[encset_name] = enc.slice(axis, start, end)

        # slice safetensors
        for set_name, weight in self.named_safetensors.items():
            if axis is not None:
                slice_indices = [slice(None, None, None)] * len(weight.shape)
                slice_indices[axis] = slice(start, end, 1)
                # pylint: disable=[unnecessary-dunder-call]
                new_weight = weight.__getitem__(tuple(slice_indices))
            else:
                new_weight = weight
            sliced_extra_info.named_safetensors[set_name] = new_weight
        return sliced_extra_info


class NamingPolicy:
    """
    Control the new generated name,
    making the new name unique
    """

    def __init__(self, prefix):
        self.name_counts = {}
        self.prefix = prefix

    def init_from_graph(self, graph: ir.Graph):
        """
        Initialize the policy,
        set proper prefix so that no existed name has the same prefix
        """
        exisiting_names = set()
        for n in graph:
            assert n.name is not None  # check for mypy
            exisiting_names.add(n.name)
            for output in n.outputs:
                assert output.name is not None  # check for mypy
                exisiting_names.add(output.name)
        for v in graph.inputs:
            assert v.name is not None  # check for mypy
            exisiting_names.add(v.name)
        for v in graph.initializers.values():
            assert v.name is not None  # check for mypy
            exisiting_names.add(v.name)

        prefix = self.prefix
        found_same_prefix = True
        while found_same_prefix:
            found_same_prefix = False
            for name in exisiting_names:
                if name.startswith(prefix):
                    prefix = "_" + prefix
                    found_same_prefix = True
                    break
        self.prefix = prefix

    def get_unique_name(self, namehint: str) -> str:
        """
        Allocate a name with namehint,
        the name is guaranteed to be unique

        Args:
            namehint: hint of the name
        Returns:
            the unique name allocated
        """
        if not namehint.startswith(self.prefix + "/"):
            name = f"{self.prefix}/{namehint}"
        else:
            name = namehint
        name_splits = name.split(".")
        if name_splits[-1].isnumeric():
            name = ".".join(name_splits[:-1])

        if name in self.name_counts:
            self.name_counts[name] += 1
            return f"{name}.{self.name_counts[name]}"

        self.name_counts[name] = 1
        return f"{name}"
