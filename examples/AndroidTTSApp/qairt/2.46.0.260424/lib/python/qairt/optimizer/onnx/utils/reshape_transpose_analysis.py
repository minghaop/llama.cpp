# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Module to provide analysis of reshape transpose sequence
"""

from collections.abc import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


class DimNode:
    """
    Represents dimensions as a dim tree,
    This class represents the node of this tree
    """

    def __init__(self, size, parent):
        self.size = size
        self.children = []
        self.parent = parent

    def __hash__(self):
        return id(self) + hash(self.size) + hash(tuple(self.children)) + hash(id(self.parent))

    def __repr__(self):
        return str(self)

    def __str__(self):
        if len(self.children) > 0:
            s = ",".join([str(x) for x in self.children])
        else:
            s = str(self.size)

        if isinstance(self.parent, DimRoot):
            return "Dim(" + s + ")"

        return s

    def contains(self, sub_node: "DimNode"):
        if sub_node is self:
            return True
        for v in self.children:
            if v is sub_node:
                return True
        for v in self.children:
            if v.contains(sub_node):
                return True
        return False

    def split_as(self, split_sizes: Iterable[int]):
        """
        Split current node into split_sizes
        the split results will be current node's children

        Args:
            split_sizes: the givin split sizes
        Returns:
            the split results (DimNode)
        """
        assert len(self.children) == 0
        assert self.size == np.prod(np.array(split_sizes))
        self.children = [DimNode(x, self) for x in split_sizes]
        return self.children

    def get_flatten_edge_nodes(self):
        """
        Get all the edge nodes in the dim tree
        Returns:
            the edge nodes
        """
        if len(self.children) == 0:
            return [self]  # edge node
        flatten_edge = []
        for x in self.children:
            flatten_edge.extend(x.get_flatten_edge_nodes())
        return flatten_edge


class OneSizeDimNode(DimNode):
    """
    Special dim node to represents one size dim
    """

    def __init__(self):
        super().__init__(1, None)


class DimRoot(DimNode):
    """
    Special dim node to represents the root of the dim tree
    """

    def __init__(self, numel):
        super().__init__(numel, None)

    def __str__(self):
        return f"[{super().__str__()}]"


class DimTree:
    """
    Represents shape in a dim tree
    so that we can track how reshape/transpose manipulate the dimensions by using thes tree
    """

    def __init__(self, shape):
        # dim node will not be replaced/removed in the reshape/transpose analysis
        # so using dim node to represent the dimension is same for any intermediate shape
        self.src_root = self.build_dim_tree(shape)
        self.curr_dims: list[list[DimNode]] = []
        for dim in self.src_root.children:
            if dim.size == 1:
                # unsqueeze dim is meaningless in this analysis
                self.curr_dims.append([OneSizeDimNode()])
            else:
                self.curr_dims.append([dim])

    def build_dim_tree(self, shape) -> DimRoot:
        """
        Build dim tree for the given tensor shape

        Args:
            shape: the shape
        Returns:
            the root node of the dim tree
        """
        numel = np.prod(np.array(shape))
        dim_root = DimRoot(numel)
        for s in shape:
            if s == 1:
                dim_node: DimNode = OneSizeDimNode()
            else:
                dim_node = DimNode(s, dim_root)
            dim_root.children.append(dim_node)
        return dim_root

    def reshape(self, shape):
        """
        Apply reshape on the dim tree, tracking how reshape manipulate the dimensions

        Args:
            shape: the output shape
        """
        curr_flattern_dims = []
        for dim in self.curr_dims:
            if isinstance(dim, list):
                curr_flattern_dims += dim
            else:
                curr_flattern_dims.append(dim)

        reshaped_dims: list[list[DimNode]] = []
        assert self.src_root.size == np.prod(np.array(shape))
        for s in shape:
            accum_s = 1
            accum_flattern_dims: list[DimNode] = []
            while accum_s < s:
                d = curr_flattern_dims[0]
                accum_s *= d.size
                if not isinstance(d, OneSizeDimNode):
                    accum_flattern_dims.append(d)
                curr_flattern_dims.pop(0)

            if len(accum_flattern_dims) == 0:
                accum_flattern_dims.append(OneSizeDimNode())
            if accum_s == s:
                reshaped_dims.append(accum_flattern_dims)
            else:
                # we need to split last dim
                accum_non_split_size = int(np.prod([x.size for x in accum_flattern_dims[:-1]]))
                if s % accum_non_split_size != 0:
                    return False
                if accum_flattern_dims[-1].size % (s // accum_non_split_size) != 0:
                    return False
                split_sizes = [
                    (s // accum_non_split_size),
                    accum_flattern_dims[-1].size // (s // accum_non_split_size),
                ]
                split_dim, remain_dim = accum_flattern_dims[-1].split_as(split_sizes)
                accum_flattern_dims[-1] = split_dim
                reshaped_dims.append(accum_flattern_dims)
                curr_flattern_dims.insert(0, remain_dim)

        # the remain unused dims should be one
        for d in curr_flattern_dims:
            assert d.size == 1

        self.curr_dims = reshaped_dims

        return True

    def transpose(self, perm):
        """
        Apply transpose on the dim tree, tracking how reshape manipulate the dimensions

        Args:
            perm: perm attribute of the transpose
        """
        assert len(self.curr_dims) == len(perm)
        self.curr_dims = [self.curr_dims[x] for x in perm]
        return True

    def map_curr_one_size_dim(self):  # pylint: disable=[too-many-locals, too-many-branches]
        """
        Map all of the current one-size-dim (after reshape/transpose)
        to the original one-size-dim

        """

        src_edge_nodes = []
        for i, d in enumerate(self.src_root.children):
            if len(d.children) > 0:
                # simplified dim_tree should be depth<=2 (including root)
                for mini_d in d.children:
                    src_edge_nodes.append(mini_d)
            else:
                src_edge_nodes.append(d)

        curr_edge_nodes = []
        for i, minidim_n_list in enumerate(self.curr_dims):
            curr_edge_nodes += minidim_n_list

        # len(flattern_curr_dims) should be equal to len(edge_nodes)
        # match one-size-dims to optimize perm if possible
        src_edge_one_dim_idx = [i for i, x in enumerate(src_edge_nodes) if x.size == 1]
        curr_edge_one_dim_idx = [i for i, x in enumerate(curr_edge_nodes) if x.size == 1]
        cost_matrix = np.zeros((len(src_edge_one_dim_idx), len(curr_edge_one_dim_idx)))
        for i, src_unsequeeze_i in enumerate(src_edge_one_dim_idx):
            for j, dst_unsqueeze_j in enumerate(curr_edge_one_dim_idx):
                cost_matrix[i, j] = abs(src_unsequeeze_i - dst_unsqueeze_j)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        paired_one_size_dims = []
        for i, j in zip(row_ind, col_ind):
            paired_one_size_dims.append(
                (src_edge_nodes[src_edge_one_dim_idx[i]], curr_edge_nodes[curr_edge_one_dim_idx[j]])
            )

        for i, d in enumerate(self.src_root.children):
            if len(d.children) > 0:
                pass
            else:
                if d.size == 1:
                    for src_edge_node, curr_edge_node in paired_one_size_dims:
                        if src_edge_node is d:
                            self.src_root.children[i] = curr_edge_node
                            break


def group_transpose(perm, in_shape):
    """
    Merge adjacent dims if they preserve same relative positions in the transpose
    e.g. [0,3,1,2] ==> [0,2,1]  original (1,2) dim is fused

    Args:
        perm: perm of the transpose
        in_shape: input shape
    Returns:
        new_perm: fused perm
        grouped_in_shape: new input shape
    """
    rank = len(perm)
    reverse_perm = [0] * rank
    for i, p in enumerate(perm):
        reverse_perm[p] = i
    perm_groups = [[perm[0]]]
    for i in range(1, rank):
        if perm[i] == perm[i - 1] + 1:
            perm_groups[-1].append(perm[i])
        else:
            perm_groups.append([perm[i]])
    # reverse_perm_groups = [[reverse_perm[x] for x in pg] for pg in perm_groups]
    # grouped_in_shape = [np.prod(np.array(in_shape)[np.array(rpg)]) for rpg in reverse_perm_groups]

    new_perm = []
    perm_groups_sorted = sorted(perm_groups, key=lambda x: x[0])
    grouped_in_shape = [np.prod(np.array(in_shape)[np.array(rpg)]) for rpg in perm_groups_sorted]
    for i, out_gp in enumerate(perm_groups):
        new_perm.append(perm_groups_sorted.index(out_gp))
    assert len(new_perm) == len(grouped_in_shape)
    return new_perm, grouped_in_shape


class ReshapeTransposeInfoSeq:
    """
    Class to represents the reshape/transpose sequence
    dosen't depend on any network framework
    """

    class BaseNodeInfo:
        """
        Basic node
        """

        def infer_shape(self, input_shape):
            """
            apply shape inference
            """
            raise NotImplementedError()

    class TransposeNodeInfo(BaseNodeInfo):
        """
        Node to represent a transpose op
        """

        def __init__(self, perm: Iterable[int]):
            self.perm: tuple = tuple(perm)
            super().__init__()

        def infer_shape(self, input_shape):
            return np.array(input_shape)[list(self.perm)].tolist()

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return False
            return self.perm == other.perm

        def __repr__(self):
            return f"Transpose({self.perm})"

        def __str__(self):
            return self.__repr__()

    class ReshapeNodeInfo(BaseNodeInfo):
        """
        Node to represent a reshape op
        """

        def __init__(self, shape):
            self.shape = shape
            super().__init__()

        def infer_shape(self, input_shape):
            return self.shape

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return False
            return self.shape == other.shape

        def __repr__(self):
            return f"Reshape({self.shape})"

        def __str__(self):
            return self.__repr__()

    def __repr__(self):
        return self.as_oneline_str()

    def pretty_print(self):
        """
        Human readable print, for debug usage
        """
        print(self.as_oneline_str())

    def as_oneline_str(self):
        """
        Print to one line string
        """
        s = f"{self.input_shape}"
        curr_shape = self.input_shape
        for n in self.seq:
            curr_shape = n.infer_shape(curr_shape)
            s = s + f"--{str(n)}-->{curr_shape}"
        return s

    def __init__(self, seq, input_shape):
        self.seq = seq
        self.input_shape = input_shape

    def build_dim_tree(self) -> tuple[list[DimTree | None], list["ReshapeTransposeInfoSeq"]]:
        """
        Build dim tree for the sequence
        """
        # it is possible that a sequence of transpose and reshape ops can not be tracked by one dim tree
        # in this situation, we will split the sequence into multiple sequences
        # for example:
        #     input = [1,4,5,2048,128]
        #     seq = [
        #         Reshape([1,4,5,1024,256]),
        #         Transpose(perm=[0,1,2,4,3]),
        #         Reshape([1,5,4,1024,256]),
        #         Transpose(perm=[0,1,2,4,3])
        #     ]
        # The corresponding results of build_dim_tree will be
        #     node_seqs = [
        #                      [Reshape([1,4,5,1024,256]), Transpose(perm=[0,1,2,4,3])],
        #                      [Reshape([1,5,4,1024,256])],
        #                      [Transpose(perm=[0,1,2,4,3])]
        #                 ]
        #     dim_trees = [dim_tree_0, None, dim_tree_2]
        #          # where dim_tree_0 is the dim_tree built with node_seqs[0]
        #          # where dim_tree_2 is the dim_tree built with node_seqs[2]

        # build dim_tree
        dim_trees: list[DimTree | None] = []
        node_seqs = []

        curr_seq: list[ReshapeTransposeInfoSeq.BaseNodeInfo] = []
        curr_dim_tree = DimTree(self.input_shape)
        curr_shape = self.input_shape
        curr_seq_input_shape = curr_shape

        i = 0
        while i < len(self.seq):
            node = self.seq[i]
            if isinstance(node, self.TransposeNodeInfo):
                status = curr_dim_tree.transpose(node.perm)
            elif isinstance(node, self.ReshapeNodeInfo):
                status = curr_dim_tree.reshape(node.shape)
            else:
                raise NotImplementedError()
            if not status:
                if len(curr_seq) > 0:
                    node_seqs.append(ReshapeTransposeInfoSeq(curr_seq, curr_seq_input_shape))
                    dim_trees.append(curr_dim_tree)
                    curr_seq = []
                # add info for node
                dim_trees.append(None)
                node_seqs.append(ReshapeTransposeInfoSeq([node], curr_shape))
                curr_shape = node_seqs[-1].infer_shape()

                curr_seq_input_shape = curr_shape
                curr_dim_tree = DimTree(curr_shape)
                i += 1
            else:
                curr_seq.append(node)
                curr_shape = node.infer_shape(curr_shape)
                i += 1
        if len(curr_seq) > 0:
            dim_trees.append(curr_dim_tree)
            node_seqs.append(ReshapeTransposeInfoSeq(curr_seq, curr_seq_input_shape))
        return dim_trees, node_seqs

    def __eq__(self, other):
        if len(self.seq) != len(other.seq):
            return False
        for i, _ in enumerate(self.seq):
            if self.seq[i] != other.seq[i]:
                return False
        if self.input_shape != other.input_shape:
            return False
        return True

    def infer_shape(self, input_shape=None):
        """
        Infer the output shape of the sequence
        """
        if input_shape is None:
            input_shape = self.input_shape
        for n in self.seq:
            if n is None:
                continue
            input_shape = n.infer_shape(input_shape)
        return input_shape

    def get_intermediate_shapes(self):
        """
        Infer all of the intermediate shapes of the sequence
        """
        intermediate_shapes = [self.input_shape]
        for n in self.seq:
            intermediate_shapes.append(n.infer_shape(intermediate_shapes[-1]))
        return intermediate_shapes

    def _get_simplified_reshape_transpose_seq(
        self,  # pylint: disable=[too-many-locals]
        src_dims: list[list[DimNode]],
        curr_dims: list[list[DimNode]],
    ):
        """
        Helper function to simplify reshape transpose sequence by dimtree
        """
        # the simplified seq will be in the form of
        # [PreReshape]->[Transpose]->[PostReshape]
        # Note: PreReshape and Transpose and PostReshape are optional

        src_edge_dims: list[tuple[DimNode]] = []
        for dims in src_dims:
            edge_d_list = []
            for x in dims:
                edge_d_list += x.get_flatten_edge_nodes()
            src_edge_dims.append(tuple(edge_d_list))

        curr_edge_dims = []
        for dims in curr_dims:
            edge_d_list = []
            for x in dims:
                edge_d_list += x.get_flatten_edge_nodes()
            curr_edge_dims.append(tuple(edge_d_list))

        src_dims_set = set(x for x in src_edge_dims)
        curr_dims_set = set(x for x in curr_edge_dims)

        same_dims_set = src_dims_set.intersection(curr_dims_set)
        simplified_src_dims = [tuple(x) for x in src_edge_dims]
        simplified_curr_dims = [tuple(x) for x in curr_edge_dims]

        # try to merge some flatten dims if there are same splitted in src/curr
        for i, dims_t in enumerate(simplified_src_dims):
            if len(dims_t) > 1 and tuple(dims_t) in same_dims_set:
                assert tuple(dims_t[0].parent.children) == dims_t
                simplified_src_dims[i] = (dims_t[0].parent,)
        for i, dims_t in enumerate(simplified_curr_dims):
            if len(dims_t) > 1 and tuple(dims_t) in same_dims_set:
                assert tuple(dims_t[0].parent.children) == dims_t
                simplified_curr_dims[i] = (dims_t[0].parent,)

        flattern_curr_dims: list[DimNode] = []
        for dims_t in simplified_curr_dims:
            flattern_curr_dims += dims_t
        flattern_src_dims: list[DimNode] = []
        for dims_t in simplified_src_dims:
            flattern_src_dims += dims_t
        dims_same_set = set(flattern_src_dims).intersection(set(flattern_curr_dims))
        # validation
        dims_diff_set = set(flattern_src_dims).union(set(flattern_curr_dims)) - dims_same_set
        for d in dims_diff_set:
            assert isinstance(d, OneSizeDimNode)

        # remove the non matched one size dim from flattern src/curr dims
        flattern_src_dims = [x for x in flattern_src_dims if x in dims_same_set]
        flattern_curr_dims = [x for x in flattern_curr_dims if x in dims_same_set]

        assert len(flattern_src_dims) == len(flattern_curr_dims)
        pre_reshape_shape = [x.size for x in flattern_src_dims]

        perm = []
        for dim in flattern_curr_dims:
            perm.append(flattern_src_dims.index(dim))

        post_reshape_shape = [int(np.prod([x.size for x in dims])) for dims in curr_dims]
        assert tuple(post_reshape_shape) == tuple(self.infer_shape(self.input_shape))

        pre_reshape = self.ReshapeNodeInfo(pre_reshape_shape)
        transpose = self.TransposeNodeInfo(perm)
        post_reshape = self.ReshapeNodeInfo(post_reshape_shape)
        return [pre_reshape, transpose, post_reshape]

    def simplify_seq(self):
        """
        Simplify seq by using DimTree
        the simplified seq will be in the standard form of
        [PreReshape]->[Transpose]->[PostReshape]
        Note: PreReshape and Transpose and PostReshape are optional
        """

        # build dim_tree
        dim_trees, node_seqs = self.build_dim_tree()
        new_seqs = []
        for dim_tree, seq in zip(dim_trees, node_seqs):
            if dim_tree is None:
                new_seqs.append(seq)
                continue

            dim_tree.map_curr_one_size_dim()

            src_dims = []
            for x in dim_tree.src_root.children:
                src_dims.append(x.get_flatten_edge_nodes())

            new_seq = seq._get_simplified_reshape_transpose_seq(src_dims, dim_tree.curr_dims)
            new_seq = seq.transform_seq_larger_than_4d_(new_seq)
            new_seq = ReshapeTransposeInfoSeq(seq.clean_seq_(new_seq), seq.input_shape)
            new_seqs.append(new_seq)

        node_seq = []
        for seq in new_seqs:
            node_seq += seq.seq

        merged_seq = ReshapeTransposeInfoSeq(node_seq, self.input_shape)
        return merged_seq

    def squeeze_larger_than_4d_seq_(self, seq_, squeeze_batch):
        # pylint: disable=[too-many-locals]
        """
        Squeeze larger than 4d sequence if possible
        """
        pre_reshape, transpose, post_reshape = seq_
        transpose_in_shape = pre_reshape.shape
        rank = len(transpose_in_shape)
        if rank <= 4:
            return seq_
        onedim_idx = [i for i, x in enumerate(transpose_in_shape) if x == 1]

        # sequeeze out the one size dim
        squeeze_out_num = max(len(pre_reshape.shape) - 4, 0)
        squeeze_out_num = min(squeeze_out_num, len(onedim_idx))
        squeeze_out_idx = onedim_idx[-squeeze_out_num:]
        if not squeeze_batch:
            # batch is special, lets keep the first dim even it's one-size
            if len(squeeze_out_idx) > 0 and 0 in squeeze_out_idx:
                squeeze_out_idx.remove(0)
        origin2squeezed_id_map = {}
        j = 0
        for i in range(len(transpose_in_shape)):
            if i not in squeeze_out_idx:
                origin2squeezed_id_map[i] = j
                j += 1
        squeezed2origin_id_map = {v: k for k, v in origin2squeezed_id_map.items()}
        new_perm = tuple(
            origin2squeezed_id_map[p] for i, p in enumerate(transpose.perm) if p in origin2squeezed_id_map
        )
        new_pre_reshape_shape = [
            transpose_in_shape[squeezed2origin_id_map[i]] for i in range(len(squeezed2origin_id_map))
        ]

        new_pre_reshape_shape = [transpose_in_shape[i] for i in range(rank) if i not in squeeze_out_idx]
        transpose.perm = new_perm
        pre_reshape.shape = new_pre_reshape_shape

        return [pre_reshape, transpose, post_reshape]

    def transform_seq_larger_than_4d_(self, seq_):
        """
        Post process the standard sequence if their dimensions are larger than 4d
        """
        seq_ = self.squeeze_larger_than_4d_seq_(seq_, False)
        pre_reshape, transpose, post_reshape = seq_
        rank = len(pre_reshape.shape)
        if rank <= 4:
            return seq_

        if transpose.perm != tuple(range(rank)):
            # if still larger than 4d,
            # then try to merge adjacent dims if they preserve same relative positions in the transpose
            new_perm, new_pre_reshape = group_transpose(transpose.perm, pre_reshape.shape)
            transpose.perm = tuple(new_perm)
            pre_reshape.shape = new_pre_reshape

            rank = len(pre_reshape.shape)

        if rank > 4:
            # still larger than 4d, try to squeeze out batch dim
            pre_reshape, transpose, post_reshape = self.squeeze_larger_than_4d_seq_(
                [pre_reshape, transpose, post_reshape], True
            )

        return [pre_reshape, transpose, post_reshape]

    def clean_seq_(self, seq):
        """
        Clean the standard sequence, remove the empty reshape/transpose
        """
        curr_shape = self.input_shape
        cleaned_seq: list[ReshapeTransposeInfoSeq.BaseNodeInfo] = []
        for node in seq:
            if isinstance(node, self.ReshapeNodeInfo):
                if tuple(curr_shape) == tuple(node.shape):
                    # reshape is useless, can be removed
                    pass
                elif len(cleaned_seq) > 0 and isinstance(cleaned_seq[-1], self.ReshapeNodeInfo):
                    # the last op is also reshape, merge it
                    cleaned_seq[-1] = node
                else:
                    cleaned_seq.append(node)
            elif isinstance(node, self.TransposeNodeInfo):
                if tuple(node.perm) == tuple(x for x in range(len(node.perm))):
                    # transpose is useless, can be removed
                    pass
                else:
                    cleaned_seq.append(node)
            else:
                assert False, "not supported yet"
            curr_shape = node.infer_shape(curr_shape)
        return cleaned_seq
