# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Tuple, Union
import warnings
from enum import Enum
import numpy as np
import torch
import torchvision
import torch.nn


########### AIMET Pro Ops #######################

class CustomLayerNorm(torch.nn.Module):
    """ Custom module for generic LayerNorm """
    def __init__(self, input_shape: list, axes: list, eps: float):
        super().__init__()
        self.input_shape = input_shape
        self.axes = axes
        self.eps = eps
        self.normalized_shape = list(input_shape[a] for a in axes)
        self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor):
        """
        Forward pass routine for custom LayerNorm
        """
        # The first permutation reorders the tensor's axes, ensuring that the axes specified in the QNN operation
        # are positioned at the last dimensions for PyTorch LayerNorm to consume
        # The second permute operation is the inverse of the first permute operation so that
        # the output tensor has the same data format as the input tensor
        permute_1_dims = [i for i in range(len(self.input_shape)) if i not in self.axes] + self.axes
        permute_2_dims = list(np.argsort(permute_1_dims))

        x = x.permute(dims=permute_1_dims)
        x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(dims=permute_2_dims)

        return x


class IndexSelect(torch.nn.Module):
    """ Custom module for IndexSelect with multiple indexes """

    # pylint: disable=unused-argument
    @staticmethod
    def forward(input_tensor: torch.Tensor, dim: int, indices: Union[torch.IntTensor, torch.LongTensor]) -> torch.Tensor:
        """ Custom forward function for IndexSelect(gather) op to handle multi dimension index values"""
        data = input_tensor
        axis = dim

        dim_size = data.shape[axis]
        original_shape = indices.shape
        indices = indices.reshape(-1)

        temp = tuple(data.shape)
        new_shape = list(temp[:axis]) + list(original_shape) + list(temp[axis+1:])

        for idx, index in enumerate(indices):
            if index < 0:
                indices[idx] = dim_size + index
        original_dtype = data.dtype
        z = torch.index_select(data.to(torch.float32), axis, indices.to(torch.int64))

        return z.reshape(*new_shape).to(original_dtype)


class NonZero(torch.nn.Module):
    """Custom module for a NonZero op"""
    @staticmethod
    def forward(tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for NonZero op
        """
        actual_value = torch.nonzero(tensor)
        n_repeat = torch.numel(tensor) - actual_value.shape[0]
        return torch.cat((actual_value, actual_value[-1].repeat(n_repeat, 1)), 0)


class Stack(torch.nn.Module):
    """
    Custom module for stack
    """
    def __init__(self, axis: int = 0):
        super().__init__()
        self._axis = axis

    def forward(self, *inputs) -> torch.Tensor:
        """
        Forward function routine for stack
        """
        return torch.stack(inputs, dim=self._axis)


class UnBind(torch.nn.Module):
    """
    Custom module for unbind
    """
    def __init__(self, axis: int = 0):
        super().__init__()
        self._axis = axis

    def forward(self, x) -> Tuple[torch.Tensor]:
        """
        Forward function routine for unbind
        """
        return torch.unbind(x, dim=self._axis)


class SpaceToBatch(torch.nn.Module):
    """ Custom module for TF/Keras based SpaceToBatch for 4D input tensor (NCHW) """

    def __init__(self, block_shape: list, pad_amount: list):
        super().__init__()
        self.block_shape = block_shape
        self.pad_amount = pad_amount if pad_amount is not None else [[0, 0], [0, 0]]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for SpaceToBatch
        """
        b, c, h, w = list(inputs.shape)
        # Constraints for block_shape
        assert len(self.block_shape) == 2, 'Invalid block_shape, must be of shape [2] with format [block_height, block_width]'
        assert self.block_shape[0] >= 1 and self.block_shape[1] >= 1, 'Invalid block_shape, elements must be >=1'
        # Constraints for pad_amount
        assert len(self.pad_amount) == 2 and len(self.pad_amount[0]) == 2, ('Invalid paddings, must be of shape [2,2] '
                                                                            'with format [[pad_top, pad_bottom], [pad_left, pad_right]]')
        # Input constraints
        assert (h+sum(self.pad_amount[0])) % self.block_shape[0] == 0 and (w + sum(self.pad_amount[1])) % self.block_shape[0] == 0, \
            'Input Constraints not satisfied'
        # STEP - 1
        padded = torch.nn.functional.pad(inputs, self.pad_amount[1] + self.pad_amount[0], "constant", 0)
        padded_shape = padded.shape
        # STEP - 2
        reshaped_padded = torch.reshape(padded, [b, c, padded_shape[2]//self.block_shape[0], self.block_shape[0],
                                                 padded_shape[3]//self.block_shape[1], self.block_shape[1]])
        # STEP - 3
        permuted_reshaped_padded = torch.permute(reshaped_padded, (3, 5, 0, 1, 2, 4))
        # STEP - 4
        output_shape = [b*self.block_shape[0]*self.block_shape[1], c, padded_shape[2]//self.block_shape[0],
                        padded_shape[3]//self.block_shape[1]]
        output = torch.reshape(permuted_reshaped_padded, output_shape)

        # Output constraints
        assert inputs.dtype == output.dtype, 'Dtypes of Input and Output are not matching'
        return output


class BatchToSpace(torch.nn.Module):
    """ Custom module for TF/Keras based BatchToSpace for 4D input tensor (NCHW) """

    def __init__(self, block_shape: list, crops: list):
        super().__init__()
        self.block_shape = block_shape
        self.crops = crops if crops is not None else [[0, 0], [0, 0]]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for BatchToSpace
        """
        b, c, h, w = list(inputs.shape)
        # Constraints for block_shape
        assert len(self.block_shape) == 2, 'Invalid block_shape, must be of shape [2] with format [block_height, block_width]'
        assert self.block_shape[0] >= 1 and self.block_shape[1] >= 1, 'Invalid block_shape, elements must be >=1'
        # Constraints for crops
        assert len(self.crops) == 2 and len(self.crops[0]) == 2, ('Invalid crops, must be of shape [2,2] with '
                                                                  'format [[crop_top, crop_bottom], [crop_left, crop_right]]')
        # Input constraints
        assert b % (self.block_shape[0] * self.block_shape[1]) == 0, 'Input Constraints not satisfied'
        assert self.crops[0][0] + self.crops[0][1] <= self.block_shape[0] * h, 'Input Constraints not satisfied'
        assert self.crops[1][0] + self.crops[1][1] <= self.block_shape[1] * w, 'Input Constraints not satisfied'
        # STEP - 1:
        reshaped = torch.reshape(inputs, self.block_shape + [b//(self.block_shape[0]*self.block_shape[1]), c] + list(inputs.shape)[2:])
        # STEP - 2:
        permuted = torch.permute(reshaped, (2, 3, 4, 0, 5, 1))
        # STEP - 3:
        reshaped_permuted = torch.reshape(permuted, [b//(self.block_shape[0]*self.block_shape[1]), c,
                                                     h*self.block_shape[0], w*self.block_shape[1]])
        # STEP - 4:
        output_height = h*self.block_shape[0]-self.crops[0][0]-self.crops[0][1]
        output_width = w*self.block_shape[1]-self.crops[1][0]-self.crops[1][1]
        output = reshaped_permuted[:, :, -output_height:, -output_width:]
        return output


class Moments(torch.nn.Module):
    """
    Custom module for moments
    """
    def __init__(self, axes: int, keep_dims: bool):
        super().__init__()
        self._axes = axes
        self._keep_dims = keep_dims

    def forward(self, inputs) -> tuple:
        """
        Forward function routine for moments
        """
        # correction=0 : To calculate biased variance which is default in case of tf.nn.moments
        var, mean = torch.var_mean(inputs, dim=self._axes, keepdim=self._keep_dims, correction=0)
        return mean, var


class CropAndResize(torch.nn.Module):
    """
    Custom PyTorch Module for tf.image.crop_and_resize
    Reference TF implementation:
    https://github.com/tensorflow/tensorflow/blob/v2.14.0/tensorflow/core/kernels/image/crop_and_resize_op_gpu.cu.cc
    """
    def __init__(self, resize_dims: list, interpolation_mode: int = 0, extrapolation_value: float = 0.0):
        super().__init__()
        self.resize_dims = resize_dims # resize_dims: [crop_height, crop_width]
        self.interpolation_mode = interpolation_mode
        self.extrapolation_value = extrapolation_value

    # pylint: disable=too-many-locals
    def forward(self, image: torch.Tensor, boxes: torch.Tensor, box_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass call for CropAndResize
        """
        _, _, img_height, img_width = image.shape
        result = None
        for box, img_ind in zip(boxes, box_indices):
            y1, x1, y2, x2 = box
            height_scale = float((y2 - y1) * (img_height - 1) / float(self.resize_dims[0] - 1) if self.resize_dims[0] > 1 else 0)
            width_scale = float((x2 - x1) * (img_width - 1) / float(self.resize_dims[1] - 1) if self.resize_dims[1] > 1 else 0)

            t_y = torch.arange(self.resize_dims[0]).to(boxes.device)
            in_y = y1 * (img_height - 1) + t_y * height_scale if self.resize_dims[0] > 1 else 0.5 * (y1 + y2) * (img_height - 1)
            t_x = torch.arange(self.resize_dims[1]).to(boxes.device)
            in_x = x1 * (img_width - 1) + t_x * width_scale if (self.resize_dims[1] > 1) else 0.5 * (x1 + x2) * (img_width - 1)

            in_x, in_y, mask, extrapolation = self._get_masks_for_extrapolation(in_x, in_y, img_width, img_height)

            if self.interpolation_mode == 0:
                image_crop = self._custom_bilinear_interpolation(image[img_ind], in_y, in_x)
            elif self.interpolation_mode == 1:
                image_crop = self._custom_nearest_neighbor_interpolation(image[img_ind], in_x, in_y)
            else:
                raise AssertionError("Only two interpolation modes are supported 0 (BILINEAR) or 1 (NEAREST NEIGHBOR)")

            image_crop = torch.unsqueeze(image_crop * mask + extrapolation, 0)
            result = image_crop if result is None else torch.cat((result, image_crop))

        result = torch.tensor(result).to(image.dtype).to(image.device)
        return result

    def _get_masks_for_extrapolation(self, in_x, in_y, img_width, img_height):
        """
        Check if extrapolation is needed and modify in_x, in_y accordingly
        """
        mask_x = torch.logical_or(in_x > (img_width-1), in_x < 0)
        mask_y = torch.logical_or(in_y > (img_height-1), in_y < 0)
        mask = torch.unsqueeze(mask_y, -1) | torch.unsqueeze(mask_x, -1).T
        mask = torch.where(mask, 0, 1)
        extrapolation = torch.logical_not(mask) * self.extrapolation_value
        in_x, in_y = torch.where(mask_x, 0, in_x), torch.where(mask_y, 0, in_y)
        return in_x, in_y, mask, extrapolation

    def _custom_bilinear_interpolation(self, image, in_y, in_x):
        """
        Custom bilinear interpolation to match with Tensorflow expectation
        """
        top_y_index, bottom_y_index = torch.floor(in_y).to(torch.long), torch.ceil(in_y).to(torch.long)
        left_x_index, right_x_index = torch.floor(in_x).to(torch.long), torch.ceil(in_x).to(torch.long)
        y_lerp = in_y - top_y_index
        x_lerp = in_x - left_x_index
        top_left = self._custom_index_select(image, top_y_index, left_x_index)
        top_right = self._custom_index_select(image, top_y_index, right_x_index)
        bottom_left = self._custom_index_select(image, bottom_y_index, left_x_index)
        bottom_right = self._custom_index_select(image, bottom_y_index, right_x_index)
        top = torch.lerp(top_left, top_right, x_lerp)
        bottom = torch.lerp(bottom_left, bottom_right, x_lerp)
        return torch.lerp(top, bottom, torch.unsqueeze(y_lerp, -1))

    def _custom_nearest_neighbor_interpolation(self, image, in_x, in_y):
        """
        Custom nearest neighbor interpolation to match with Tensorflow expectation
        """
        closest_y_index = torch.round(in_y).to(torch.long)
        closest_x_index = torch.round(in_x).to(torch.long)
        return self._custom_index_select(image, closest_y_index, closest_x_index)

    @staticmethod
    def _custom_index_select(a, y_ind, x_ind):
        """
        Gather tensor 'a' along height (y_ind) and width (x_ind) axes
        """
        a = torch.index_select(a, 1, y_ind)
        return torch.index_select(a, 2, x_ind)


class MultiClassNms(torch.nn.Module):
    """
    Custom module for QNN MultiClassNms, only supports Hard NMS for now
    """
    def __init__(self, iou_threshold: float, score_threshold: float = 0.0, soft_nms_sigma: float = 0.0, max_output_boxes_per_batch: int = None):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms_sigma = soft_nms_sigma
        self.max_output_boxes_per_batch = max_output_boxes_per_batch

    # pylint: disable=too-many-locals
    def forward(self, *inp):
        """
        Forward pass for MultiClassNms
        """
        batched_boxes = inp[0]  # [batch, num_boxes, 4]
        batched_scores = inp[1]  # [batch, num_boxes, num_classes]
        batched_features = ()
        num_boxes = batched_boxes.shape[1]
        num_classes = batched_scores.shape[-1]
        if len(inp) > 2:
            batched_features = inp[2:]  # tuple of feature vectors whose dimension is [batch, num_boxes, ...]

        output_classes = None
        output_boxes = None
        output_scores = None
        output_indices = []
        output_features = ()

        # Iterate through batch dimension
        for boxes, scores in zip(batched_boxes, batched_scores):
            # Repeat boxes to allow Multiclass NMS
            boxes = boxes.repeat(num_classes, 1)
            # Maintain original box indices tensor to ensure that NMS gives unique boxes
            box_ind = torch.arange(0, num_boxes, dtype=torch.int64).repeat(num_classes)
            # Flatten scores to allow Multiclass NMS
            scores = scores.transpose(1, 0).flatten()
            # Maintain classes tensor as we are flattening scores
            classes = torch.arange(0, num_classes, dtype=torch.int64).repeat_interleave(num_boxes)

            # TODO: Add support for soft_nms_sigma after converter starts supporting it
            unique_boxes, unique_box_ind, unique_scores, unique_classes = self._perform_hard_nms(boxes, box_ind, scores, classes)
            # Ensure only unique boxes are returned
            if unique_box_ind.shape != torch.unique(unique_box_ind).shape:
                warnings.warn("'torchvision.ops.batched_nms' couldn't return unique boxes")
                # batched NMS returns indices in descending order of scores. As we are performing NMS within
                # each class if the NMS output has same box multiple times (with different class labels),
                # we should pick the first one.
                unique_box_ind, ind_ = self._get_unique_first_indices(unique_box_ind)
                unique_boxes = unique_boxes[ind_]
                unique_scores = unique_scores[ind_]
                unique_classes = unique_classes[ind_]

            b = self._get_zero_filled_tensor(unique_boxes, (1, self.max_output_boxes_per_batch, 4))
            c = self._get_zero_filled_tensor(unique_classes, (1, self.max_output_boxes_per_batch, ))
            s = self._get_zero_filled_tensor(unique_scores, (1, self.max_output_boxes_per_batch, ))

            output_classes = torch.cat([output_classes, c]) if output_classes is not None else c
            output_boxes = torch.cat([output_boxes, b]) if output_boxes is not None else b
            output_scores = torch.cat([output_scores, s]) if output_scores is not None else s
            output_indices.append(unique_box_ind)

        # Iterate through features
        for batched_feature in batched_features:
            # un-batch each feature to gather output features using NMS output indices
            output_feature = [self._custom_unsqueezed_index_select(feature, ind) for ind, feature in zip(output_indices, batched_feature)]
            output_feature_shape = (1, self.max_output_boxes_per_batch, *tuple(batched_feature.shape[2:]))
            # re-batch each output feature
            output_feature = torch.concat([self._get_zero_filled_tensor(feature, output_feature_shape) for feature in output_feature])
            output_features += (output_feature,)

        return (output_boxes, output_scores, output_classes, *output_features)[:len(inp) + 1]

    @staticmethod
    def _modify_y1x1y2x2_to_x1y1x2y2(boxes):
        return boxes[:, torch.tensor([1, 0, 3, 2])]

    @staticmethod
    def _custom_unsqueezed_index_select(tensor: torch.Tensor, ind: torch.Tensor, dim: int = 1):
        return torch.index_select(tensor.unsqueeze(0), dim, ind) if ind.numel() else torch.tensor([])

    @staticmethod
    def _get_unique_first_indices(tensor: torch.Tensor):
        unique, idx, counts = torch.unique(tensor, dim=1, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        return unique, ind_sorted[cum_sum]

    @staticmethod
    def _get_zero_filled_tensor(tensor: torch.Tensor, dim: Union[Tuple, list]):
        """
        Return zero filled tensor with fixed shape, as dynamic shapes are not allowed
        """
        out = torch.zeros(*dim, dtype=tensor.dtype, device=tensor.device)
        indices = torch.arange(0, int(np.prod(tensor.shape)), 1, dtype=torch.int64, device=tensor.device)
        return out.put_(indices, tensor)

    def _perform_hard_nms(self, boxes: torch.Tensor, box_ind: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor):
        """
        Filter bounding boxes across multiple classes in descending order of score using Non-max suppression
        """
        # Filter scores using score threshold
        filtered_score_ind = (scores > self.score_threshold).nonzero()[:, 0]
        filtered_boxes = boxes[filtered_score_ind]
        filtered_box_ind = box_ind[filtered_score_ind]
        filtered_scores = scores[filtered_score_ind]
        filtered_classes = classes[filtered_score_ind]
        # Sort the scores in descending order
        sorted_scores, sorted_ind = torch.sort(filtered_scores, descending=True, stable=True)
        sorted_boxes = filtered_boxes[sorted_ind]
        sorted_box_ind = filtered_box_ind[sorted_ind]
        sorted_classes = filtered_classes[sorted_ind]
        # Use batched_nms, to allow NMS only between boxes of same class
        res_ = torchvision.ops.batched_nms(self._modify_y1x1y2x2_to_x1y1x2y2(sorted_boxes), sorted_scores, sorted_classes, self.iou_threshold)
        if res_.shape[0] > self.max_output_boxes_per_batch:
            res_ = res_[:self.max_output_boxes_per_batch]
        return sorted_boxes[res_], sorted_box_ind[res_], sorted_scores[res_], sorted_classes[res_]


class CustomPReLU(torch.nn.Module):
    """
    Custom PRelu module to allow weights with dims > 1
    """
    def __init__(self, weight_shape: list):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full(weight_shape, fill_value=0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for custom PReLU
        """
        return torch.where(x >= 0, x, 0.0) + self.weight * torch.where(x < 0, x, 0.0)


class RNNDirection(Enum):
    """
    Enum for RNN direction
    """
    FORWARD = 0
    BACKWARD = 1


class CustomStatefulLSTM(torch.nn.Module):
    """
    Custom StatefulLSTM Pytorch implementation for HTP Quantization
    """
    def __init__(self, time_steps, num_units, batch_size, input_size, output_size, time_major, direction):
        super().__init__()
        self.time_steps = time_steps
        self.num_units = num_units
        self.input_size = input_size
        self.output_size = output_size
        self.time_major = time_major
        self.batch_size = batch_size
        self.direction = direction
        self._register_hidden_state_buffers()
        self._register_params()

    def _register_hidden_state_buffers(self):
        """
        Register Hidden state buffers that stay alive until the model is moved out of memory
        """
        # h_t, c_t need NOT be stored in state_dict (non-persistent buffers), but need to stored in memory
        self.register_buffer("h_t", torch.zeros([self.batch_size, self.output_size], dtype=torch.float32), persistent=False)
        self.register_buffer("c_t", torch.zeros([self.batch_size, self.num_units], dtype=torch.float32), persistent=False)

    def _reset_buffers_to_zero(self):
        """
        Reset hidden and cell state buffers to zeros
        """
        self.h_t = torch.zeros([self.batch_size, self.output_size], dtype=torch.float32).to(self.h_t.device)
        self.c_t = torch.zeros([self.batch_size, self.num_units], dtype=torch.float32).to(self.c_t.device)

    def _register_params(self):
        """
        Register gate params (weights and biases)
        """
        # Define weights
        # Input to gate weights
        self.register_parameter('w_xf', torch.nn.Parameter(torch.ones([self.num_units, self.input_size], dtype=torch.float32)))
        self.register_parameter('w_xc', torch.nn.Parameter(torch.ones([self.num_units, self.input_size], dtype=torch.float32)))
        self.register_parameter('w_xo', torch.nn.Parameter(torch.ones([self.num_units, self.input_size], dtype=torch.float32)))
        self.register_parameter('w_xi', torch.nn.Parameter(torch.ones([self.num_units, self.input_size], dtype=torch.float32)))

        # Hidden state to gate weights
        self.register_parameter('w_hf', torch.nn.Parameter(torch.ones([self.num_units, self.output_size], dtype=torch.float32)))
        self.register_parameter('w_hc', torch.nn.Parameter(torch.ones([self.num_units, self.output_size], dtype=torch.float32)))
        self.register_parameter('w_ho', torch.nn.Parameter(torch.ones([self.num_units, self.output_size], dtype=torch.float32)))
        self.register_parameter('w_hi', torch.nn.Parameter(torch.ones([self.num_units, self.output_size], dtype=torch.float32)))

        # Gate Biases
        self.register_parameter('b_f', torch.nn.Parameter(torch.zeros([self.num_units], dtype=torch.float32)))
        self.register_parameter('b_c', torch.nn.Parameter(torch.zeros([self.num_units], dtype=torch.float32)))
        self.register_parameter('b_o', torch.nn.Parameter(torch.zeros([self.num_units], dtype=torch.float32)))
        self.register_parameter('b_i', torch.nn.Parameter(torch.zeros([self.num_units], dtype=torch.float32)))

    def forward(self, x: torch.Tensor, h_0: torch.Tensor, c_0: torch.Tensor, reset: bool):
        """
        Forward pass for Stateful LSTM
        :param x: input tensor
        :param h_0: in[10] from LSTM OpDef
        :param c_0: in[11] from LSTM OpDef
        :param reset: Flag to determine whether to reset hidden state buffers with h_0 and c_0 respectively
        """
        # For a 3D tensor, if the tensor is not time major, make it time major internally.
        # Finally, the shape of x will be (time_steps, batch_size, input_size)
        if len(x.shape) == 3 and not self.time_major:
            x = x.permute(1, 0, 2)

        # For Backward LSTM, flip inputs along 'time_steps' dim
        if self.direction == RNNDirection.BACKWARD:
            x = torch.flip(x, [0])

        # If reset is True, we will reset the hidden state and cell state buffers
        if reset:
            self.h_t = h_0
            self.c_t = c_0

        H_t = []
        for t in range(self.time_steps):
            # Calculate intermediate gate pre-activations
            matmul_i = x[t] @ self.w_xi.T + self.h_t @ self.w_hi.T + self.b_i
            matmul_f = x[t] @ self.w_xf.T + self.h_t @ self.w_hf.T + self.b_f
            matmul_c = x[t] @ self.w_xc.T + self.h_t @ self.w_hc.T + self.b_c
            matmul_o = x[t] @ self.w_xo.T + self.h_t @ self.w_ho.T + self.b_o

            # Apply activations to intermediate gate pre-activations
            i_t = torch.nn.functional.sigmoid(matmul_i)
            f_t = torch.nn.functional.sigmoid(matmul_f)
            g_t = torch.nn.functional.tanh(matmul_c)
            o_t = torch.nn.functional.sigmoid(matmul_o)

            # Update hidden and cell state buffers
            self.c_t = f_t * self.c_t + i_t * g_t
            self.h_t = o_t * torch.nn.functional.tanh(self.c_t)
            H_t.append(self.h_t)

        # Stack the outputs across time steps
        H_t = torch.stack(H_t)

        # For Backward Lstm, flip outputs along 'time_steps' dim,
        # as they are stacked from last time step to the first
        if self.direction == RNNDirection.BACKWARD:
            H_t = torch.flip(H_t, [0])

        # If input dim order is not time major, restore original order
        if len(x.shape) == 3 and not self.time_major:
            H_t = H_t.permute(1, 0, 2)

        # If input is 2D, return the output too in 2D
        if len(x.shape) == 2:
            H_t = torch.squeeze(H_t, 0)

        return H_t, self.c_t, self.h_t, matmul_i, matmul_f, matmul_c, matmul_o


class CustomStatefulGru(torch.nn.Module):
    """
   Custom StatefulGru Pytorch implementation for HTP Quantization
   """
    def __init__(self, seq_length, batch_size, input_size, hidden_size, time_major, direction, linear_before_reset):
        super().__init__()
        self.seq_length = seq_length
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.time_major = time_major
        self.direction = direction
        self.linear_before_reset = linear_before_reset
        self._register_hidden_state_buffers()
        self._register_params()

    def _register_hidden_state_buffers(self):
        """
        Register hidden state buffer that stay alive until the model is moved out of memory
        """
        # h_t need NOT be stored in state_dict (non-persistent buffer), but need to stored in memory
        self.register_buffer("h_t", torch.zeros([1, self.batch_size, self.hidden_size], dtype=torch.float32), persistent=False)

    def _reset_buffers_to_zero(self):
        """
        Reset hidden state buffer to zero
        """
        self.h_t = torch.zeros([1, self.batch_size, self.hidden_size], dtype=torch.float32).to(self.h_t.device)

    def _register_params(self):
        """
        Register gate params (weights and biases)
        """
        # Define weights
        # Input to gate weights
        self.register_parameter('w_xz', torch.nn.Parameter(torch.ones([self.hidden_size, self.input_size], dtype=torch.float32)))
        self.register_parameter('w_xr', torch.nn.Parameter(torch.ones([self.hidden_size, self.input_size], dtype=torch.float32)))
        self.register_parameter('w_xn', torch.nn.Parameter(torch.ones([self.hidden_size, self.input_size], dtype=torch.float32)))

        # Hidden state to gate weights
        self.register_parameter('w_hz', torch.nn.Parameter(torch.ones([self.hidden_size, self.hidden_size], dtype=torch.float32)))
        self.register_parameter('w_hr', torch.nn.Parameter(torch.ones([self.hidden_size, self.hidden_size], dtype=torch.float32)))
        self.register_parameter('w_hn', torch.nn.Parameter(torch.ones([self.hidden_size, self.hidden_size], dtype=torch.float32)))

        # Gate Biases
        self.register_parameter('b_xz', torch.nn.Parameter(torch.zeros([self.hidden_size], dtype=torch.float32)))
        self.register_parameter('b_xr', torch.nn.Parameter(torch.zeros([self.hidden_size], dtype=torch.float32)))
        self.register_parameter('b_xn', torch.nn.Parameter(torch.zeros([self.hidden_size], dtype=torch.float32)))
        self.register_parameter('b_hz', torch.nn.Parameter(torch.zeros([self.hidden_size], dtype=torch.float32)))
        self.register_parameter('b_hr', torch.nn.Parameter(torch.zeros([self.hidden_size], dtype=torch.float32)))
        self.register_parameter('b_hn', torch.nn.Parameter(torch.zeros([self.hidden_size], dtype=torch.float32)))

    def forward(self, X: torch.Tensor, h_0: torch.Tensor, reset: bool):
        """
        Forward pass for Stateful Gru
        :param X: input tensor
        :param h_0: initial_h from Gru OpDef
        :param reset: Flag to determine whether to reset hidden state buffers with h_0 and c_0 respectively
        """
        # For a 3D tensor, if the tensor is not time major, make it time major internally.
        # Finally, the shape of x will be (time_steps, batch_size, input_size)
        if len(X.shape) == 3 and not self.time_major:
            X = X.permute(1, 0, 2)

        # For Backward LSTM, flip inputs along 'time_steps' dim
        if self.direction == RNNDirection.BACKWARD:
            X = torch.flip(X, [0])

        # If reset is True, we will reset the hidden state and cell state buffers
        if reset:
            self.h_t = h_0

        Y = []
        for t in range(self.seq_length):
            matmul_xr = X[t] @ self.w_xr.T + self.b_xr
            matmul_xz = X[t] @ self.w_xz.T + self.b_xz
            matmul_hr = self.h_t @ self.w_hr.T + self.b_hr
            matmul_hz = self.h_t @ self.w_hz.T + self.b_hz

            r_t = torch.nn.functional.sigmoid(matmul_xr + matmul_hr)
            z_t = torch.nn.functional.sigmoid(matmul_xz + matmul_hz)

            matmul_xn = X[t] @ self.w_xn.T + self.b_xn
            if self.linear_before_reset:
                matmul_hn = r_t * (self.h_t @ self.w_hn.T + self.b_hn)
            else:
                matmul_hn = (r_t * self.h_t) @ self.w_hn.T + self.b_hn
            n_t = torch.nn.functional.tanh(matmul_xn + matmul_hn)

            # Update hidden state buffer
            self.h_t = (1-z_t) * n_t + z_t * self.h_t
            Y.append(self.h_t)

        # Stack the outputs across all time steps
        Y = torch.concat(Y)

        # For Backward Gru, flip outputs along 'time_steps' dim,
        # as they are stacked from last time step to the first
        if self.direction == RNNDirection.BACKWARD:
            Y = torch.flip(Y, [0])

        # If input dim order is not time major, restore the dim order
        if len(X.shape) == 3 and not self.time_major:
            Y = Y.permute(1, 0, 2)

        if len(X.shape) == 2:
            Y = torch.squeeze(Y, 0)

        return Y, self.h_t


class NonBlockingBuffer(torch.nn.Module):
    def __init__(self, buffer_shape, buffer_dim, buffer_padding, mode, stride):
        super().__init__()
        self.buffer_shape = buffer_shape
        self.buffer_dim = buffer_dim
        self.buffer_padding = buffer_padding
        self.buffer_size = buffer_shape[buffer_dim]
        self.mode = mode
        self.is_buffer_full = False
        # Initialise number of valid frames to buffer_padding
        self.num_valid_frames = buffer_padding
        assert mode in [1, 2], "Only non-blocking buffers are supported currently in MPP!"
        self.stride = stride
        assert self.buffer_shape[self.buffer_dim] == self.buffer_size
        self._register_non_blocking_buffer()

    def _register_non_blocking_buffer(self):
        """
        Register non-blocking buffer that stays alive until the model is moved out of memory
        """
        # need NOT be stored in state_dict (non-persistent buffer), but need to stored in memory
        self.register_buffer("buffer", torch.zeros(self.buffer_shape, dtype=torch.float32), persistent=False)

    def _reset_buffers_to_zero(self):
        """
        Reset hidden state buffer to zero
        """
        self.buffer = torch.zeros(self.buffer_shape, dtype=torch.float32).to(self.buffer.device)

    def _return_buffer_output(self):
        if self.num_valid_frames == self.buffer_size:
            self.is_buffer_full = True
        return self.buffer

    def forward(self, inp_frame: torch.Tensor, reset: bool):
        frame_shape = list(inp_frame.shape)
        num_input_frames = int(frame_shape[self.buffer_dim])
        assert self.buffer_size % num_input_frames == 0, (f"Number of frames in input {num_input_frames} must "
                                                          f"divide buffer_size {self.buffer_shape[self.buffer_dim]}")
        assert self.stride % num_input_frames == 0, (f"Stride{self.stride} must be divisible by number "
                                                     f"of frames in the input {num_input_frames}")

        # If reset is True, reset the buffer to zero
        if reset:
            self.buffer = torch.zeros(self.buffer_shape, dtype=torch.float32).to(self.buffer.device)
            self.num_valid_frames = self.buffer_padding
            self.is_buffer_full = False

        # If the buffer is full and num_valid_frames == buffer_size, remove
        # the oldest frames of size stride and append the incoming input frame
        # Note: Removal and population behavior is same for all modes
        if self.num_valid_frames == self.buffer_size and self.is_buffer_full:
            _, rem_frames = torch.split(self.buffer, [self.stride, self.buffer_size-self.stride], self.buffer_dim)
            # Zero Padding is only required when stride size is not same as input size
            if self.stride == num_input_frames:
                self.buffer = torch.concat([rem_frames, inp_frame], dim=self.buffer_dim)
            else:
                num_padding_frames = self.stride - num_input_frames
                pad_shape = frame_shape
                pad_shape[self.buffer_dim] = num_padding_frames
                padded_frame = torch.zeros(pad_shape, dtype=torch.float32).to(self.buffer.device)
                self.buffer = torch.concat([rem_frames, inp_frame, padded_frame], dim=self.buffer_dim)
                self.num_valid_frames = self.buffer_size - self.stride + num_input_frames
        else:
            num_padding_frames = self.buffer_size - (self.num_valid_frames + num_input_frames)
            frames = []
            if num_padding_frames != 0:
                pad_shape = frame_shape
                pad_shape[self.buffer_dim] = num_padding_frames
                padded_frame = torch.zeros(pad_shape, dtype=torch.float32).to(self.buffer.device)
                frames.append(padded_frame)

            split_sections = [self.num_valid_frames, self.buffer_size-self.num_valid_frames] \
                if (self.mode==1 or self.is_buffer_full) else [self.buffer_size-self.num_valid_frames, self.num_valid_frames]

            split_frames = torch.split(self.buffer, split_sections, self.buffer_dim)

            # If the buffer is full once or num_valid_frames != buffer_size, take the valid frames and
            # concat it with the incoming frame and zero pad the rest of the buffer if the buffer is not full.
            # Note: Population behavior is same for all modes.
            if self.is_buffer_full:
                # [split_frames[0], inp_frame, padded_frame]
                frames.insert(0, inp_frame)
                frames.insert(0, split_frames[0])
            # If the buffer is not full and then add the incoming frame based on the mode
            else:
                if self.mode == 1:
                    # NB left: [split_frames[0], inp_frame, padded_frame]
                    frames.insert(0, inp_frame)
                    frames.insert(0, split_frames[0])
                elif self.mode == 2:
                    # NB right: [padded_frame, split_frames[1], inp_frame]
                    frames.extend([split_frames[1], inp_frame])

            self.buffer = torch.concat(frames, dim=self.buffer_dim)
            self.num_valid_frames += num_input_frames

        return self._return_buffer_output()
