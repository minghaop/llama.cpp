# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

""" File contains QcModule for the LLaMa style LLM """

import math
from qti.aisw.graph_gen.mad.lib import op, module
from qti.aisw.graph_gen.mad.lib.module import ModuleMarker
from string import Template

class LLaMa(module.QcModule):
    """
    LLaMa model implementation as a QcModule.
    """
    def __init__(self, config):
        """
        Initializes the LLaMa model.

        :param config: Configuration object for the LLaMa model.
        """
        self.split_size = config.num_hidden_layers
        if config.split > 1:
            self.split_size = math.ceil(config.num_hidden_layers / config.split)
        assert config.split <= 9, (
            "Invalid split count: Genie supports a maximum of 9 splits "
            f"(received {config.split})."
        )
        self.last_layer_index = config.num_hidden_layers - 1
        # Ensure the actual number of splits matches the requested value.
        # Example: num_hidden_layers = 4, splits = 3 → split_size = 2 (ceil(4 / 2)).
        # This results in only two valid splits ([0–1], [2–3]); the third would be empty.
        # Using a strict '<' avoids creating such redundant final splits.
        assert (config.split - 1) * self.split_size < config.num_hidden_layers, (
            "Invalid split count: the last split would be empty. "
            "Reduce number_splits by one."
        )

        self.num_hidden_layers = config.num_hidden_layers
        self.embedding = op.QcEmbedding(config.vocab_size, config.hidden_size)
        if config.sha:
            self.decoder_blocks = [DecoderBlockSHA(config)  for _ in range(config.num_hidden_layers)]
        else:
            self.decoder_blocks = [DecoderBlockMHA(config)  for _ in range(config.num_hidden_layers)]

        self.output_norm = op.QcRMSNormOp(config.rms_norm_eps, 2, config.hidden_size, False)
        self.pre_reshape = op.QcReshape([config.batch, config.seq_length, 1, config.hidden_size])
        self.pre_tranpose = op.QcTranspose([0,2, 1,3])

        self.lm_head = op.QcConv(config.hidden_size, config.vocab_size, bias=False)
        self.post_transpose = op.QcTranspose([0,2, 1,3])
        self.logits = op.QcReshape([config.batch, config.seq_length, config.vocab_size])

    def forward(self, input_ids, position_ids_sin, position_ids_cos, attention_mask, past_key_in, past_value_in):
        """
        Forward pass for the LLaMa model.

        :param input_ids: Input token IDs.
        :param position_ids_sin: Sine components of positional embeddings.
        :param position_ids_cos: Cosine components of positional embeddings.
        :param attention_mask: Mask for attention.
        :param past_key_in: Input past key states.
        :param past_value_in: Input past value states.
        :return: Tuple containing output logits and updated past key/value states.
        """
        out = self.embedding(input_ids)
        outs = []

        for i in range(self.num_hidden_layers):
            out, past_key_out, past_value_out = self.decoder_blocks[i](out, position_ids_sin, position_ids_cos, attention_mask, past_key_in[i], past_value_in[i])
            outs.append(past_key_out)
            outs.append(past_value_out)
            # To achieve an n split, n–1 split markers are required. Therefore, the split marker at the 
            # end of the final hidden layer is omitted.
            if i==self.last_layer_index:
                break
            if (i+1) % self.split_size == 0:
                ModuleMarker.split_at(out)

        out = self.output_norm(out)
        out = self.pre_reshape(out)
        out = self.pre_tranpose(out)
        out = self.lm_head(out)
        out = self.post_transpose(out)
        out = self.logits(out)
        return out, *outs

    @staticmethod
    def parameter_encoding_map(num_attention_heads:int , num_key_value_heads: int) -> dict:
        """
        Returns a mapping template dict to map parameter tensor names from MHA graph to SHA graph.
        Mapping depends on the DecoderBlock definition used in the model.

        :param num_attention_heads: Number of attention heads in the graph
        :param num_key_value_heads: Number of key-value heads in the graph
        :return: Mapping from MHA name to a list of tuple or tuple containing the template and the head counts.
        """

        return {
            "k_proj": (Template("k_heads.$num.k_proj"), num_key_value_heads),
            "v_proj": (Template("v_heads.$num"), num_key_value_heads),
            "q_proj": (Template("q_heads.$num.q_proj"), num_attention_heads),
        }

    @staticmethod
    def activation_encoding_map(num_attention_heads:int , num_key_value_heads: int) -> dict:
        """
        Returns a mapping template dict to map activation tensor names from MHA graph to SHA graph.
        Mapping depends on the DecoderBlock definition used in the model.

        :param num_attention_heads: Number of attention heads in the graph
        :param num_key_value_heads: Number of key-value heads in the graph
        :return: Mapping from MHA name to a list of tuple or tuple containing the template and the head counts.
        """

        return {
            # Key Projection and past key concatenation
            "k_concat_output_0":(Template("k_concat.${num}_output_0"), num_key_value_heads),
            "k_proj_output_0": (Template("k_heads.${num}.k_proj_output_0"), num_key_value_heads),

            # Query Projection
            "q_proj_output_0":(Template("q_heads.${num}.q_proj_output_0"), num_attention_heads),

            # Key Rope Entries
            "rope_k.add_s1_sin_s2_cos_output_0":(Template("k_heads.${num}.rope_k.add_s1_sin_s2_cos_output_0"), num_key_value_heads),
            "rope_k.concat_s1_s2_output_0":(Template("k_heads.${num}.rope_k.concat_s1_s2_output_0"), num_key_value_heads),
            "rope_k.mul_s1_cos_output_0":(Template("k_heads.${num}.rope_k.mul_s1_cos_output_0"), num_key_value_heads),
            "rope_k.mul_s1_sin_output_0":(Template("k_heads.${num}.rope_k.mul_s1_sin_output_0"), num_key_value_heads),
            "rope_k.mul_s2_cos_output_0":(Template("k_heads.${num}.rope_k.mul_s2_cos_output_0"), num_key_value_heads),
            "rope_k.mul_s2_sin_output_0":(Template("k_heads.${num}.rope_k.mul_s2_sin_output_0"), num_key_value_heads),
            "rope_k.sub_s1_os_s2_sin_output_0":(Template("k_heads.${num}.rope_k.sub_s1_os_s2_sin_output_0"), num_key_value_heads),

            # Query Rope Entries
            "rope_q.add_s1_sin_s2_cos_output_0":(Template("q_heads.${num}.rope_q.add_s1_sin_s2_cos_output_0"), num_attention_heads),
            "rope_q.concat_s1_s2_output_0":(Template("q_heads.${num}.rope_q.concat_s1_s2_output_0"), num_attention_heads),
            "rope_q.mul_s1_cos_output_0":(Template("q_heads.${num}.rope_q.mul_s1_cos_output_0"), num_attention_heads),
            "rope_q.mul_s1_sin_output_0":(Template("q_heads.${num}.rope_q.mul_s1_sin_output_0"), num_attention_heads),
            "rope_q.mul_s2_cos_output_0":(Template("q_heads.${num}.rope_q.mul_s2_cos_output_0"), num_attention_heads),
            "rope_q.mul_s2_sin_output_0":(Template("q_heads.${num}.rope_q.mul_s2_sin_output_0"), num_attention_heads),
            "rope_q.sub_s1_os_s2_sin_output_0":(Template("q_heads.${num}.rope_q.sub_s1_os_s2_sin_output_0"), num_attention_heads),

            # Attention entries
            "spda.add_mask_output_0":(Template("spda_heads.${num}.add_mask_output_0"), num_attention_heads),
            "spda.self.div_value":(Template("spda_heads.${num}.self.div_value"), num_attention_heads),
            "spda.div_output_0":(Template("spda_heads.${num}.div_output_0"), num_attention_heads),
            "spda.qkt_output_0":(Template("spda_heads.${num}.qkt_output_0"), num_attention_heads),
            "spda.softmax_output_0":(Template("spda_heads.${num}.softmax_output_0"), num_attention_heads),
            "spda.score_output_0":[(Template("spda_heads.${num}.score_output_0"), num_attention_heads), (Template("spda_concat_output_0"), 1)],

            # value Projection and past value concatenation
            "v_concat_output_0":(Template("v_concat.${num}_output_0"), num_key_value_heads),
            "v_proj_output_0":[(Template("v_heads.${num}_output_0"), num_key_value_heads), (Template("past_value_out_output_0"), 1)],
        }


class ScaledDotProductAttention(module.QcModule):
    """
    Scaled Dot-Product Attention mechanism.
    """
    def __init__(self, head_dim):
        """
        Initializes the ScaledDotProductAttention module.

        :param head_dim[int]: Dimension of each attention head.
        """
        self.div_value = math.sqrt(head_dim)
        self.qkt = op.QcMatMul()
        self.div = op.QcDiv()
        self.add_mask = op.QcAdd()
        self.softmax = op.QcSoftmax(dim = 3)
        self.score = op.QcMatMul()

    def forward(self, q, k, v, mask):
        """
        Forward pass for Scaled Dot-Product Attention.

        :param q: Query tensor.
        :param k: Key tensor.
        :param v: Value tensor.
        :param mask: Attention mask.
        :return: Output of the attention mechanism.
        """
        out = self.qkt(q, k)
        out = self.div(out, self.div_value)
        out = self.add_mask(out, mask)
        out = self.softmax(out)
        out = self.score(out, v)
        return out


class DecoderBlockMHA(module.QcModule):
    """
    Multi-Head Attention (MHA) Decoder Block.
    """
    def __init__(self, config):
        """
        Initializes the DecoderBlockMHA module.

        :param config: Configuration object for the decoder block.
        """
        self.pre_norm = op.QcRMSNormOp(config.rms_norm_eps,2, config.hidden_size, False)
        self.pre_reshape = op.QcReshape([config.batch, config.seq_length, 1, config.hidden_size])
        self.pre_tranpose = op.QcTranspose([0,2, 1,3])

        self.k_proj = op.QcConv(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.k_post_transpose = op.QcTranspose([0,2, 1,3])
        self.k_reshape = op.QcReshape([config.batch, config.seq_length, config.num_key_value_heads, config.head_dim])
        self.k_transpose = op.QcTranspose([0,2,1,3])
        self.rope_k = PositionalEnbeddingRoPE(config)
        self.past_key_out = op.QcTranspose([0, 1, 3, 2])
        self.k_concat = op.QcConcat(dim = 3)
        self.k_unsqueeze = op.QcReshape([config.batch,  config.num_key_value_heads, 1, config.head_dim,  config.max_position_embeddings])
        self.k_expand = op.QcExpand([config.batch,  config.num_key_value_heads, config.num_attention_heads//config.num_key_value_heads, config.head_dim,  config.max_position_embeddings])
        self.k_current_reshape = op.QcReshape([config.batch,  config.num_attention_heads, config.head_dim,  config.max_position_embeddings])

        self.v_proj = op.QcConv(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.v_post_transpose = op.QcTranspose([0,2, 1,3])
        self.v_reshape = op.QcReshape([config.batch, config.seq_length, config.num_key_value_heads, config.head_dim])
        self.past_value_out = op.QcTranspose([0, 2, 1, 3])
        self.v_concat = op.QcConcat(dim = 2)
        self.v_unsqueeze = op.QcReshape([config.batch,  config.num_key_value_heads, 1, config.max_position_embeddings, config.head_dim])
        self.v_expand = op.QcExpand([config.batch,  config.num_key_value_heads, config.num_attention_heads//config.num_key_value_heads, config.max_position_embeddings, config.head_dim])
        self.v_current_reshape = op.QcReshape([config.batch,  config.num_attention_heads, config.max_position_embeddings, config.head_dim])

        self.q_proj = op.QcConv(config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.attention_bias)
        self.q_post_transpose = op.QcTranspose([0,2, 1,3])
        self.q_reshape = op.QcReshape([config.batch, config.seq_length, config.num_attention_heads, config.head_dim])
        self.q_transpose = op.QcTranspose([0,2,1,3])
        self.rope_q = PositionalEnbeddingRoPE(config)

        self.spda = ScaledDotProductAttention(config.head_dim)
        self.attn_tranapose = op.QcTranspose([0,2,1,3])
        self.attn_reshape = op.QcReshape([config.batch, config.seq_length, 1, config.num_attention_heads * config.head_dim]) # B, S, 1, H
        self.o_proj_pre_transpose = op.QcTranspose([0,2, 1,3])   # B, 1, S, H, sequence length is treated as width
        self.o_proj = op.QcConv(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        self.o_proj_post_transpose = op.QcTranspose([0,2, 1,3])  # B, S, 1, H

        self.post_reshape = op.QcReshape([config.batch, config.seq_length, config.hidden_size])
        self.residual_post_attn = op.QcAdd()

        self.post_norm = op.QcRMSNormOp(config.rms_norm_eps,2, config.hidden_size, False)
        self.mlp = MLP(config)
        self.residual_post_mlp = op.QcAdd()

    def forward(self, input, pos_ids_sin, pos_ids_cos, attention_mask, past_key, past_value):
        """
        Forward pass for the Multi-Head Attention Decoder Block.

        :param input: Input tensor.
        :param pos_ids_sin: Sine components of positional embeddings.
        :param pos_ids_cos: Cosine components of positional embeddings.
        :param attention_mask: Attention mask.
        :param past_key: Past key states.
        :param past_value: Past value states.
        :return: Tuple containing output tensor, updated key, and updated value.
        """
        out = self.pre_norm(input)
        out = self.pre_reshape(out)
        out = self.pre_tranpose(out)

        k_out = self.k_proj(out)
        k_out = self.k_post_transpose(k_out)
        k_out = self.k_reshape(k_out)
        k_out = self.k_transpose(k_out)
        k_out = self.rope_k(k_out, pos_ids_sin, pos_ids_cos)
        k_out = self.past_key_out(k_out)
        key_out = k_out # For past key output

        k_out = self.k_concat(past_key, k_out)
        k_out = self.k_unsqueeze(k_out)
        k_out = self.k_expand(k_out)
        k_out = self.k_current_reshape(k_out)


        v_out = self.v_proj(out)
        v_out = self.v_post_transpose(v_out)
        v_out = self.v_reshape(v_out)
        v_out = self.past_value_out(v_out)
        value_out = v_out # For past value output

        v_out = self.v_concat(past_value, v_out)
        v_out = self.v_unsqueeze(v_out)
        v_out = self.v_expand(v_out)
        v_out = self.v_current_reshape(v_out)


        q_out = self.q_proj(out)
        q_out = self.q_post_transpose(q_out)
        q_out = self.q_reshape(q_out)
        q_out = self.q_transpose(q_out)
        q_out = self.rope_q(q_out, pos_ids_sin, pos_ids_cos)

        attn = self.spda(q_out, k_out, v_out, attention_mask)
        attn = self.attn_tranapose(attn)
        attn = self.attn_reshape(attn)
        attn = self.o_proj_pre_transpose(attn)
        attn = self.o_proj(attn)
        attn = self.o_proj_post_transpose(attn)
        attn = self.post_reshape(attn)


        attn = self.residual_post_attn(input, attn)

        out = self.post_norm(attn)
        out = self.mlp(out)
        out = self.residual_post_mlp(attn, out)

        return out, key_out, value_out



class SHAKeyHead(module.QcModule):
    """
    Key head for Shared-Head Attention (SHA).
    """
    def __init__(self,config):
        """
        Initializes the SHAKeyHead module.

        :param config: Configuration object.
        """
        self.k_proj = op.QcConv(config.hidden_size, 1 * config.head_dim, bias=config.attention_bias)
        self.rope_k = PositionalEnbeddingRoPE(config)
        self.k_rope_transpose = op.QcTranspose([0, 1, 3, 2])

    def forward(self, input, pos_ids_sin, pos_ids_cos):
        """
        Forward pass for the SHA Key Head.

        :param input: Input tensor.
        :param pos_ids_sin: Sine components of positional embeddings.
        :param pos_ids_cos: Cosine components of positional embeddings.
        :return: Output key tensor.
        """
        k_out = self.k_proj(input)
        k_out = self.rope_k(k_out, pos_ids_sin, pos_ids_cos)
        k_out = self.k_rope_transpose(k_out)
        return k_out


class SHAQueryHead(module.QcModule):
    """
    Query head for Shared-Head Attention (SHA).
    """
    def __init__(self,config):
        """
        Initializes the SHAQueryHead module.

        :param config: Configuration object.
        """
        self.q_proj = op.QcConv(config.hidden_size, 1 * config.head_dim, bias=config.attention_bias)
        self.rope_q = PositionalEnbeddingRoPE(config)

    def forward(self, input, pos_ids_sin, pos_ids_cos):
        """
        Forward pass for the SHA Query Head.

        :param input: Input tensor.
        :param pos_ids_sin: Sine components of positional embeddings.
        :param pos_ids_cos: Cosine components of positional embeddings.
        :return: Output query tensor.
        """
        q_out = self.q_proj(input)
        q_out = self.rope_q(q_out, pos_ids_sin, pos_ids_cos)
        return q_out


class DecoderBlockSHA(module.QcModule):
    """
    Shared-Head Attention (SHA) Decoder Block.
    """
    def __init__(self, config):
        """
        Initializes the DecoderBlockSHA module.

        :param config: Configuration object for the decoder block.
        """
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        assert self.num_attention_heads%self.num_key_value_heads == 0
        self.head_ratio = self.num_attention_heads // self.num_key_value_heads

        self.pre_norm = op.QcRMSNormOp(config.rms_norm_eps,2, config.hidden_size, False)
        self.pre_reshape = op.QcReshape([config.batch, config.seq_length, 1, config.hidden_size])
        self.pre_tranpose = op.QcTranspose([0,2, 1,3])

        # Key and Values are sliced at Batch dimension as they were concatenated at batch dimension instead of kv_head dimension due to some HTP optimization.
        # As we are using batch size one it is mathematically equivalent.
        assert 1 == config.batch
        slice_dim = 0 # Later change it to 0 for better HTP performance
        self.past_key_in = [op.QcStridedSliceOp(axes=[slice_dim], slice_ranges=[[index, index + 1, 1]]) for index in range(config.num_key_value_heads)]
        self.past_value_in = [op.QcStridedSliceOp(axes=[slice_dim], slice_ranges=[[index, index + 1, 1]]) for index in range(config.num_key_value_heads)]

        self.k_heads = [SHAKeyHead(config) for _ in range(config.num_key_value_heads)]
        self.k_concat = [op.QcConcat(dim=3) for _ in range(config.num_key_value_heads)]
        self.past_key_out = op.QcConcat(dim = slice_dim)

        self.v_heads = [op.QcConv(config.hidden_size, 1 * config.head_dim, bias=config.attention_bias) for _ in range(config.num_key_value_heads)]
        self.v_concat = [op.QcConcat(dim=2) for _ in range(config.num_key_value_heads)]
        self.past_value_out = op.QcConcat(dim = slice_dim)

        self.q_heads = [SHAQueryHead(config) for _ in range(config.num_attention_heads)]

        self.spda_heads = [ScaledDotProductAttention(config.head_dim) for _ in range(config.num_attention_heads)]
        self.spda_concat = op.QcConcat(dim = 3)

        # Output is already concatenated at the head dim so already in B, 1, S, H format so no need to perform extra reshape and transpose.
        self.o_proj = op.QcConv(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        self.o_proj_post_transpose = op.QcTranspose([0,2, 1,3])  # B, S, 1, H

        self.post_reshape = op.QcReshape([config.batch, config.seq_length, config.hidden_size])
        self.residual_post_attn = op.QcAdd()

        self.post_norm = op.QcRMSNormOp(config.rms_norm_eps,2, config.hidden_size, False)
        self.mlp = MLP(config)
        self.residual_post_mlp = op.QcAdd()

    def forward(self, input, pos_ids_sin, pos_ids_cos, attention_mask, past_key, past_value):
        """
        Forward pass for the Shared-Head Attention Decoder Block.

        :param input: Input tensor.
        :param pos_ids_sin: Sine components of positional embeddings.
        :param pos_ids_cos: Cosine components of positional embeddings.
        :param attention_mask: Attention mask.
        :param past_key: Past key states.
        :param past_value: Past value states.
        :return: Tuple containing output tensor, updated past key, and updated past value.
        """
        out = self.pre_norm(input)
        out = self.pre_reshape(out)
        out = self.pre_tranpose(out)

        k_complete = []
        v_complete = []

        k_slices = []
        v_slices = []

        for i in range(self.num_key_value_heads):
            past_key_in = self.past_key_in[i](past_key)
            past_value_in = self.past_value_in[i](past_value)

            k_head_out = self.k_heads[i](out, pos_ids_sin, pos_ids_cos)
            v_head_out = self.v_heads[i](out)
            k_slices.append(k_head_out)
            v_slices.append(v_head_out)

            k_concat_out = self.k_concat[i](past_key_in, k_head_out)
            v_concat_out = self.v_concat[i](past_value_in, v_head_out)

            k_complete.append(k_concat_out)
            v_complete.append(v_concat_out)

        past_key_out = self.past_key_out(*k_slices)
        past_value_out = self.past_value_out(*v_slices)

        attn_outs = []
        for i in range(self.num_attention_heads):
            q_head_out = self.q_heads[i](out, pos_ids_sin, pos_ids_cos)
            attn_slice = self.spda_heads[i](q_head_out, k_complete[i//self.head_ratio], v_complete[i//self.head_ratio], attention_mask)
            attn_outs.append(attn_slice)


        attn = self.spda_concat(*attn_outs)

        attn = self.o_proj(attn)
        attn = self.o_proj_post_transpose(attn)
        attn = self.post_reshape(attn)

        attn = self.residual_post_attn(input, attn)

        out = self.post_norm(attn)
        out = self.mlp(out)
        out = self.residual_post_mlp(attn, out)

        return out, past_key_out, past_value_out


class MLP(module.QcModule):
    """
    Multi-Layer Perceptron (MLP) module.
    """
    def __init__(self, config):
        """
        Initializes the MLP module.

        :param config: Configuration object.
        """
        self.pre_reshape = op.QcReshape([config.batch, config.seq_length, 1, config.hidden_size])
        self.pre_tranpose = op.QcTranspose([0,2, 1,3])
        self.gate_proj = op.QcConv(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.act = SiLU()
        self.up_proj = op.QcConv(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.act_fn = op.QcMul()
        self.down_proj = op.QcConv(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.post_transpose = op.QcTranspose([0,2, 1,3])
        self.post_reshape = op.QcReshape([config.batch, config.seq_length, config.hidden_size])

    def forward(self, input):
        """
        Forward pass for the MLP module.

        :param input: Input tensor.
        :return: Output tensor.
        """
        input = self.pre_reshape(input)
        input = self.pre_tranpose(input)
        gate_out = self.gate_proj(input)
        gate_out = self.act(gate_out)
        up_out = self.up_proj(input)
        down_out = self.act_fn(up_out, gate_out)
        down_out = self.down_proj(down_out)
        down_out = self.post_transpose(down_out)
        down_out = self.post_reshape(down_out)
        return down_out

class SiLU(module.QcModule):
    """
    Sigmoid Linear Unit (SiLU) activation function.
    """
    def __init__(self):
        """
        Initializes the SiLU activation function.
        """
        self.sig = op.QcSigmoid()
        self.mul = op.QcMul()

    def forward(self, input):
        """
        Forward pass for the SiLU activation function.

        :param input: Input tensor.
        :return: Output tensor.
        """
        out = self.sig(input)
        out = self.mul(input, out)
        return out

class PositionalEnbeddingRoPE(module.QcModule):
    """
    Rotary Positional Embedding (RoPE) module.
    """
    def __init__(self, config):
        """
        Initializes the PositionalEnbeddingRoPE module.

        :param config: Configuration object.
        """
        self.slice_1 = op.QcStridedSliceOp(axes=[3], slice_ranges=[[0, config.head_dim//2, 1]])
        self.slice_2 = op.QcStridedSliceOp(axes=[3], slice_ranges=[[config.head_dim//2, config.head_dim, 1]])

        self.mul_s1_cos = op.QcMul()
        self.mul_s2_sin = op.QcMul()
        self.sub_s1_os_s2_sin = op.QcSub()

        self.mul_s1_sin = op.QcMul()
        self.mul_s2_cos = op.QcMul()
        self.add_s1_sin_s2_cos = op.QcAdd()

        self.concat_s1_s2 = op.QcConcat(dim = 3)

    def forward(self, input, pos_ids_sin, pos_ids_cos):
        """
        Forward pass for the PositionalEnbeddingRoPE module.

        :param input: Input tensor.
        :param pos_ids_sin: Sine components of positional embeddings.
        :param pos_ids_cos: Cosine components of positional embeddings.
        :return: Output tensor with rotary positional embeddings applied.
        """
        s1 = self.slice_1(input)
        s2 = self.slice_2(input)

        s1_cos = self.mul_s1_cos(s1, pos_ids_cos)
        s2_sin = self.mul_s2_sin(s2, pos_ids_sin)
        sub_op = self.sub_s1_os_s2_sin(s1_cos, s2_sin)

        s1_sin = self.mul_s1_sin(s1, pos_ids_sin)
        s2_cos = self.mul_s2_cos(s2, pos_ids_cos)
        add_op = self.add_s1_sin_s2_cos(s1_sin, s2_cos)

        concat_s1_s2 = self.concat_s1_s2(sub_op, add_op)
        return concat_s1_s2
