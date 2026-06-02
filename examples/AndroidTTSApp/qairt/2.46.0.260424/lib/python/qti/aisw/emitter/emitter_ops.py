# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Custom PyTorch Modules for QNN ops """

import copy
import torch.nn
import logging
logger = logging.getLogger('TorchEmitter')
from qti.aisw.emitter.op_definition import *

try:
    from aimet_torch.v2.nn.true_quant import QuantizationMixin
    from aimet_torch.v2.quantization.affine import QuantizeDequantize
    from aimet_torch import translation_mapping
    # These Op mappings are needed for AIMET to recognize corresponding backend Op for a given pytorch layer
    translation_mapping.aimet_op_to_backend_op_name_map[CustomLayerNorm] = "LayerNorm"
    translation_mapping.aimet_op_to_backend_op_name_map[CustomPReLU] = "Prelu"
    translation_mapping.aimet_op_to_backend_op_name_map[SpaceToBatch] = "SpaceToBatch"
    translation_mapping.aimet_op_to_backend_op_name_map[BatchToSpace] = "BatchToSpace"
    translation_mapping.aimet_op_to_backend_op_name_map[NonZero] = "NonZero"
    translation_mapping.aimet_op_to_backend_op_name_map[CropAndResize] = "CropAndResize"
    translation_mapping.aimet_op_to_backend_op_name_map[MultiClassNms] = "MultiClassNms"
    translation_mapping.aimet_op_to_backend_op_name_map[CustomStatefulLSTM] = "Lstm"
    translation_mapping.aimet_op_to_backend_op_name_map[CustomStatefulGru] = "Gru"
    translation_mapping.aimet_op_to_backend_op_name_map[NonBlockingBuffer] = "Buffer"

    from aimet_torch import onnx_utils
    # These Op mappings are needed for AIMET to recognize corresponding onnx
    # op name for a given pytorch layer, used by QuantsimConfigurator.
    onnx_utils.map_torch_types_to_onnx[CustomLayerNorm] = ["LayerNormalization"]
    onnx_utils.map_torch_types_to_onnx[CropAndResize] = ["CropAndResize"]
    onnx_utils.map_torch_types_to_onnx[SpaceToBatch] = ["SpaceToBatch"]
    onnx_utils.map_torch_types_to_onnx[BatchToSpace] = ["BatchToSpace"]
    onnx_utils.map_torch_types_to_onnx[NonZero] = ["NonZero"]
    onnx_utils.map_torch_types_to_onnx[CustomPReLU] = ["PRelu"]
    onnx_utils.map_torch_types_to_onnx[CustomStatefulLSTM] = ['LSTM']
    onnx_utils.map_torch_types_to_onnx[CustomStatefulGru] = ['GRU']
    onnx_utils.map_torch_types_to_onnx[NonBlockingBuffer] = ['Buffer']

    from aimet_torch.meta import connectedgraph
    connectedgraph.SKIP_LIST_FOR_SUBGRAPH_TRACE = (
        CustomLayerNorm, CustomPReLU, SpaceToBatch,
        BatchToSpace, CropAndResize, MultiClassNms,
        CustomStatefulLSTM,
        CustomStatefulGru,
        NonBlockingBuffer,
        *connectedgraph.SKIP_LIST_FOR_SUBGRAPH_TRACE
    )

    from aimet_torch.quantsim_config import quantsim_config
    quantsim_config.MAP_PYTORCH_PARAM_NAME_TO_QUANTSIM_NAME.update({
        "b_i": "bias", "b_f": "bias", "b_c": "bias", "b_o": "bias",
        "b_xz": "bias", "b_xr": "bias", "b_xn": "bias",
        "b_hz": "bias", "b_hr": "bias", "b_hn": "bias"
    })

    def _map_quantizer_to_tensors(tensors, quantizer):
        maybe_quantize = lambda t: quantizer(t) if quantizer and t.is_floating_point() else t
        return torch.utils._pytree.tree_map(maybe_quantize, tensors) # pylint: disable=protected-access


    def _unary_forward(self: QuantizationMixin, *args, **kwargs) -> torch.Tensor: # pylint: disable=missing-function-docstring
        x, *others = args

        if isinstance(x, torch.Tensor) and x.is_floating_point() and self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters(): # pylint: disable=protected-access
            output = super(type(self), self).forward(x, *others, **kwargs)

        if isinstance(output, torch.Tensor) and output.is_floating_point() and self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output

    # pylint: disable=too-many-ancestors
    @QuantizationMixin.implements(CustomLayerNorm)
    class QuantizedCustomLayerNorm(QuantizationMixin, CustomLayerNorm):
        """ Quantized class definition for CustomLayerNorm """
        forward = _unary_forward

    FakeQuantizedCustomLayerNorm = QuantizedCustomLayerNorm


    @QuantizationMixin.implements(IndexSelect)
    class QuantizedIndexSelect(QuantizationMixin, IndexSelect):
        """ Quantized class definition for IndexSelect """
        forward = _unary_forward

    FakeQuantizedIndexSelect = QuantizedIndexSelect


    @QuantizationMixin.implements(BatchToSpace)
    class QuantizedBatchToSpace(QuantizationMixin, BatchToSpace):
        """ Quantized class definition for BatchToSpace """
        forward = _unary_forward

    FakeQuantizedBatchToSpace = QuantizedBatchToSpace


    @QuantizationMixin.implements(SpaceToBatch)
    class QuantizedSpaceToBatch(QuantizationMixin, SpaceToBatch):
        """ Quantized class definition for SpaceToBatch """
        forward = _unary_forward

    FakeQuantizedSpaceToBatch = QuantizedSpaceToBatch


    @QuantizationMixin.implements(CropAndResize)
    class QuantizedCropAndResize(QuantizationMixin, CropAndResize):
        """ Quantized class definition for CropAndResize """
        forward = _unary_forward

    FakeQuantizedCropAndResize = QuantizedCropAndResize


    @QuantizationMixin.implements(UnBind)
    class QuantizedUnBind(QuantizationMixin, UnBind):
        """ Quantized class definition for UnBind """
        _num_outputs: int

        def __quant_init__(self):
            super().__quant_init__()
            self._num_outputs = None

        # pylint: disable=arguments-differ
        # pylint: disable=too-many-function-args
        def export_output_encodings(self, encoding_version: str):
            output_encodings = super().export_output_encodings(encoding_version)
            if self._num_outputs is None:
                raise RuntimeError("Cannot infer number of output tensors without first executing `self.forward`.")
            # Create separate encoding objects to avoid overriding of attributes added/updated later while exporting encodings
            return [copy.deepcopy(encoding) for encoding in output_encodings * self._num_outputs]

        def forward(self, x) -> Tuple[torch.Tensor]: # pylint: disable=arguments-differ
            """ Quantized forward definition for UnBind """

            x = _map_quantizer_to_tensors(x, self.input_quantizers[0])

            outputs = super().forward(x)

            self._num_outputs = len(outputs)

            return _map_quantizer_to_tensors(outputs, self.output_quantizers[0])

    FakeQuantizedUnBind = QuantizedUnBind


    @QuantizationMixin.implements(NonZero)
    class QuantizedNonZero(QuantizationMixin, NonZero):
        """ Quantized class definition for NonZero """

        def __quant_init__(self):
            super().__quant_init__()
            self.output_quantizers = torch.nn.ModuleList([])

        def forward(self, tensor: torch.Tensor) -> torch.Tensor: # pylint: disable=arguments-differ
            """ Quantized forward definition for NonZero """
            tensor = _map_quantizer_to_tensors(tensor, self.input_quantizers[0])

            return super().forward(tensor)

    FakeQuantizedNonZero = QuantizedNonZero


    @QuantizationMixin.implements(Moments)
    class QuantizedMoments(QuantizationMixin, Moments):
        """ Quantized class definition for Moments """

        def __quant_init__(self):
            super().__quant_init__()
            self.output_quantizers = torch.nn.ModuleList([None, None])

        def forward(self, inputs) -> Tuple: # pylint: disable=arguments-differ
            """ Quantized forward definition for Moments """
            inputs = _map_quantizer_to_tensors(inputs, self.input_quantizers[0])

            mean, var = super().forward(inputs)

            if self.output_quantizers[0]:
                mean = self.output_quantizers[0](mean)

            if self.output_quantizers[1]:
                var = self.output_quantizers[1](var)

            return mean, var

    FakeQuantizedMoments = QuantizedMoments


    @QuantizationMixin.implements(Stack)
    class QuantizedStack(QuantizationMixin, Stack): # pylint: disable=too-many-ancestors
        """
        Quantized class definition for Stack.
        """
        _num_inputs: int

        def __quant_init__(self):
            super().__quant_init__()
            self._num_inputs = 1

        # pylint: disable=arguments-differ
        # pylint: disable=too-many-function-args
        def export_input_encodings(self, encoding_version: str):
            input_encodings = super().export_input_encodings(encoding_version)
            # Create separate encoding objects to avoid overriding of attributes added/updated later while exporting encodings
            return [copy.deepcopy(encoding) for encoding in input_encodings * self._num_inputs]

        def forward(self, *inputs) -> torch.Tensor: # pylint: disable=arguments-differ
            """
            Quantized forward impl for Stack.
            """
            self._num_inputs = len(inputs)

            inputs = _map_quantizer_to_tensors(inputs, self.input_quantizers[0])

            output = super().forward(*inputs)

            if output.is_floating_point() and self.output_quantizers[0]:
                output = self.output_quantizers[0](output)

            return output

    FakeQuantizedStack = QuantizedStack


    @QuantizationMixin.implements(MultiClassNms)
    class QuantizedMultiClassNms(QuantizationMixin, MultiClassNms):
        """
        Quantized class definition for MultiClassNms.
        """
        def __quant_init__(self):
            super().__quant_init__()
            # Apply one input quantizer to all *batched_features
            self.input_quantizers = torch.nn.ModuleList([None])
            # Apply output_quantizer[0] to scores, output_quantizer[1] to all output_features
            self.output_quantizers = torch.nn.ModuleList([None, None])

        def forward(self, *inp): # pylint: disable=arguments-differ
            """
            Quantized forward impl for Stack.
            """
            batched_features = ()
            if len(inp) > 2:
                batched_features = inp[2:]

            batched_features = _map_quantizer_to_tensors(batched_features, self.input_quantizers[0])

            output_boxes, output_scores, output_classes, *output_features = super().forward(*inp[0:2], *batched_features)

            if self.output_quantizers[0]:
                output_scores = self.output_quantizers[0](output_scores)

            output_features = _map_quantizer_to_tensors(output_features, self.output_quantizers[1])

            return output_boxes, output_scores, output_classes, *output_features

    FakeQuantizedMultiClassNms = QuantizedMultiClassNms

    @QuantizationMixin.implements(CustomPReLU)
    class QuantizedCustomPReLU(QuantizationMixin, CustomPReLU):
        """ Quantized class definition for CustomPReLU """
        forward = _unary_forward


    @QuantizationMixin.implements(CustomStatefulLSTM)
    class QuantizedCustomStatefulLSTM(QuantizationMixin, CustomStatefulLSTM):
        def __quant_init__(self):
            super().__quant_init__()
            self.param_quantizers = torch.nn.ModuleDict()
            for gate_suffix in ['i', 'o', 'c', 'f']:
                for inp_prefix in ['x', 'h']:
                    self.param_quantizers[f"w_{inp_prefix}{gate_suffix}"] = None
                self.param_quantizers[f"b_{gate_suffix}"] = None

            self.input_quantizers = torch.nn.ModuleList([None, None, None, None])
            self.output_quantizers = torch.nn.ModuleList([None, None, None, None, None, None, None])

        def forward(self, x, h_0, c_0, reset):

            x = _map_quantizer_to_tensors(x, self.input_quantizers[0])
            h_0 = _map_quantizer_to_tensors(h_0, self.input_quantizers[1])
            c_0 = _map_quantizer_to_tensors(c_0, self.input_quantizers[2])

            with self._patch_quantized_parameters():
                #####################################################################################
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

                    # Collect intermediate gate output quant statistics at every time step
                    matmul_i = _map_quantizer_to_tensors(matmul_i, self.output_quantizers[3])
                    matmul_f = _map_quantizer_to_tensors(matmul_f, self.output_quantizers[4])
                    matmul_c = _map_quantizer_to_tensors(matmul_c, self.output_quantizers[5])
                    matmul_o = _map_quantizer_to_tensors(matmul_o, self.output_quantizers[6])

                    # Apply activations to intermediate gate pre-activations
                    i_t = torch.nn.functional.sigmoid(matmul_i)
                    f_t = torch.nn.functional.sigmoid(matmul_f)
                    g_t = torch.nn.functional.tanh(matmul_c)
                    o_t = torch.nn.functional.sigmoid(matmul_o)

                    # Update hidden and cell state buffers
                    self.c_t = f_t * self.c_t + i_t * g_t
                    self.h_t = o_t * torch.nn.functional.tanh(self.c_t)
                    # Collect hidden and cell state quant statistics at every time step
                    self.c_t = _map_quantizer_to_tensors(self.c_t, self.output_quantizers[1])
                    self.h_t = _map_quantizer_to_tensors(self.h_t, self.output_quantizers[2])

                    H_t.append(self.h_t)

                # Stack the outputs across time steps
                H_t = torch.stack(H_t)

                # For Backward Lstm, flip outputs along 'time_steps' dim,
                # as they are stacked from last time step to the first
                if self.direction == RNNDirection.BACKWARD:
                    H_t = torch.flip(H_t, [0])

                # If input dim order is not time major, restore the dim order
                if len(x.shape) == 3 and not self.time_major:
                    H_t = H_t.permute(1, 0, 2)

                # If input is 2D, return 2D output
                if len(x.shape) == 2:
                    H_t = torch.squeeze(H_t, 0)
                ##########################################################################################################

            H_t = _map_quantizer_to_tensors(H_t, self.output_quantizers[0])

            return H_t, self.c_t, self.h_t, matmul_i, matmul_f, matmul_c, matmul_o

        def _create_int32_bias_quantizer(self, *args):
            """
            Create int32 bias quantizers
            """
            for gate_suffix in ['i', 'o', 'f', 'c']:
                bias_name = f"b_{gate_suffix}"
                bias = getattr(self, bias_name)
                if bias is not None:
                    wx_qtzr = self.param_quantizers[f"w_x{gate_suffix}"]
                    wh_qtzr = self.param_quantizers[f"w_h{gate_suffix}"]
                    encoding_shape = bias.shape if isinstance(wx_qtzr, list) and isinstance(wh_qtzr, list) else ()

                    qmin = -(2**31)
                    qmax = 2**31 - 1

                    bias_qtzr = QuantizeDequantize(
                        shape=encoding_shape, qmin=qmin, qmax=qmax, symmetric=True
                    )
                    bias_qtzr.to(dtype=bias.dtype, device=bias.device)
                    self.param_quantizers[bias_name] = bias_qtzr

                    if not bias_qtzr.is_initialized():
                        # Failed to derive bias encodings analytically from input and weight encodings.
                        # Fall back to statistical bias encoding calibration.
                        # This should be avoided as much as possible
                        with bias_qtzr.compute_encodings():
                            _ = bias_qtzr(bias)

    @QuantizationMixin.implements(CustomStatefulGru)
    class QuantizedCustomStatefulGru(QuantizationMixin, CustomStatefulGru):
        def __quant_init__(self):
            super().__quant_init__()
            self.param_quantizers = torch.nn.ModuleDict()
            for gate_suffix in ['z', 'r', 'n']:
                for inp_prefix in ['x', 'h']:
                    self.param_quantizers[f"w_{inp_prefix}{gate_suffix}"] = None
                    self.param_quantizers[f"b_{inp_prefix}{gate_suffix}"] = None

            self.input_quantizers = torch.nn.ModuleList([None, None, None])
            self.output_quantizers = torch.nn.ModuleList([None, None])

        def forward(self, X, h_0, reset):

            X = _map_quantizer_to_tensors(X, self.input_quantizers[0])
            h_0 = _map_quantizer_to_tensors(h_0, self.input_quantizers[1])

            with self._patch_quantized_parameters():
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

                    self.h_t = (1-z_t) * n_t + z_t * self.h_t
                    # Collect hidden state quant statistics at every time step
                    self.h_t = _map_quantizer_to_tensors(self.h_t, self.output_quantizers[1])
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

            Y = _map_quantizer_to_tensors(Y, self.output_quantizers[0])

            return Y, self.h_t

        def _create_int32_bias_quantizer(self, *args):
            """
            Create int32 bias quantizers
            """
            for gate_suffix in ['z', 'r', 'n']:
                for inp_prefix in ['x', 'h']:
                    bias_name = f"b_{inp_prefix}{gate_suffix}"
                    bias = getattr(self, bias_name)
                    if bias is not None:
                        w_qtzr = self.param_quantizers[f"w_{inp_prefix}{gate_suffix}"]
                        encoding_shape = bias.shape if isinstance(w_qtzr, list) else ()

                        qmin = -(2**31)
                        qmax = 2**31 - 1

                        bias_qtzr = QuantizeDequantize(
                            shape=encoding_shape, qmin=qmin, qmax=qmax, symmetric=True
                        )
                        bias_qtzr.to(dtype=bias.dtype, device=bias.device)
                        self.param_quantizers[bias_name] = bias_qtzr

                        if not bias_qtzr.is_initialized():
                            # Failed to derive bias encodings analytically from input and weight encodings.
                            # Fall back to statistical bias encoding calibration.
                            # This should be avoided as much as possible
                            with bias_qtzr.compute_encodings():
                                _ = bias_qtzr(bias)

    @QuantizationMixin.implements(NonBlockingBuffer)
    class QuantizedNonBlockingBuffer(QuantizationMixin, NonBlockingBuffer):
        def __quant_init__(self):
            super().__quant_init__()
            self.input_quantizers = torch.nn.ModuleList([None, None])
            self.output_quantizers = torch.nn.ModuleList([None])

        def forward(self, inp_frame, reset):
            inp_frame = _map_quantizer_to_tensors(inp_frame, self.input_quantizers[0])
            output = super().forward(inp_frame, reset)
            return _map_quantizer_to_tensors(output, self.output_quantizers[0])


except ModuleNotFoundError:
    logger.warning('Model Loaded without quantization node definition!')
