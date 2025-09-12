#pragma once

#include "llama.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-hparams.h"
#include "llama-memory.h"
#include "llama-vocab.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_ubatch;
struct llama_model_loader;

// available models
enum llm_type {
    LLM_TYPE_UNKNOWN,
    LLM_TYPE_14M,
    LLM_TYPE_17M,
    LLM_TYPE_22M,
    LLM_TYPE_33M,
    LLM_TYPE_60M,
    LLM_TYPE_70M,
    LLM_TYPE_80M,
    LLM_TYPE_109M,
    LLM_TYPE_137M,
    LLM_TYPE_160M,
    LLM_TYPE_190M,
    LLM_TYPE_220M,
    LLM_TYPE_250M,
    LLM_TYPE_256M,
    LLM_TYPE_270M,
    LLM_TYPE_335M,
    LLM_TYPE_350M,
    LLM_TYPE_410M,
    LLM_TYPE_450M,
    LLM_TYPE_475M,
    LLM_TYPE_700M,
    LLM_TYPE_770M,
    LLM_TYPE_780M,
    LLM_TYPE_0_3B,
    LLM_TYPE_0_5B,
    LLM_TYPE_0_6B,
    LLM_TYPE_1B,
    LLM_TYPE_1_2B,
    LLM_TYPE_1_3B,
    LLM_TYPE_1_4B,
    LLM_TYPE_1_5B,
    LLM_TYPE_1_6B,
    LLM_TYPE_1_7B,
    LLM_TYPE_1_8B,
    LLM_TYPE_2B,
    LLM_TYPE_2_8B,
    LLM_TYPE_2_9B,
    LLM_TYPE_3B,
    LLM_TYPE_4B,
    LLM_TYPE_6B,
    LLM_TYPE_6_9B,
    LLM_TYPE_7B,
    LLM_TYPE_8B,
    LLM_TYPE_9B,
    LLM_TYPE_11B,
    LLM_TYPE_12B,
    LLM_TYPE_13B,
    LLM_TYPE_14B,
    LLM_TYPE_15B,
    LLM_TYPE_16B,
    LLM_TYPE_20B,
    LLM_TYPE_27B,
    LLM_TYPE_30B,
    LLM_TYPE_32B,
    LLM_TYPE_34B,
    LLM_TYPE_35B,
    LLM_TYPE_40B,
    LLM_TYPE_65B,
    LLM_TYPE_70B,
    LLM_TYPE_142B,
    LLM_TYPE_236B,
    LLM_TYPE_290B,
    LLM_TYPE_314B,
    LLM_TYPE_405B,
    LLM_TYPE_671B,
    LLM_TYPE_SMALL,
    LLM_TYPE_MEDIUM,
    LLM_TYPE_LARGE,
    LLM_TYPE_XL,
    LLM_TYPE_A1_7B,
    LLM_TYPE_A2_7B,
    LLM_TYPE_8x7B,
    LLM_TYPE_8x22B,
    LLM_TYPE_16x12B,
    LLM_TYPE_16x3_8B,
    LLM_TYPE_10B_128x3_66B,
    LLM_TYPE_57B_A14B,
    LLM_TYPE_17B_16E, // llama4 Scout
    LLM_TYPE_17B_128E, // llama4 Maverick
    LLM_TYPE_A13B,
    LLM_TYPE_30B_A3B,
    LLM_TYPE_235B_A22B,
    LLM_TYPE_E2B,
    LLM_TYPE_E4B,
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type);

struct llama_layer_posnet {
    // resnet
    struct ggml_tensor * norm1   = nullptr;
    struct ggml_tensor * norm1_b = nullptr;

    struct ggml_tensor * conv1   = nullptr;
    struct ggml_tensor * conv1_b = nullptr;

    struct ggml_tensor * norm2   = nullptr;
    struct ggml_tensor * norm2_b = nullptr;

    struct ggml_tensor * conv2   = nullptr;
    struct ggml_tensor * conv2_b = nullptr;

    // attention
    struct ggml_tensor * attn_norm   = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;

    struct ggml_tensor * attn_q   = nullptr;
    struct ggml_tensor * attn_q_b = nullptr;

    struct ggml_tensor * attn_k   = nullptr;
    struct ggml_tensor * attn_k_b = nullptr;

    struct ggml_tensor * attn_v   = nullptr;
    struct ggml_tensor * attn_v_b = nullptr;

    struct ggml_tensor * attn_o   = nullptr;
    struct ggml_tensor * attn_o_b = nullptr;

    // normalize
    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;
};

struct llama_layer_convnext {
    struct ggml_tensor * dw   = nullptr;
    struct ggml_tensor * dw_b = nullptr;

    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;

    struct ggml_tensor * pw1   = nullptr;
    struct ggml_tensor * pw1_b = nullptr;

    struct ggml_tensor * pw2   = nullptr;
    struct ggml_tensor * pw2_b = nullptr;

    struct ggml_tensor * gamma = nullptr;
};

struct llama_layer_shortconv {
    struct ggml_tensor * in_proj  = nullptr;
    struct ggml_tensor * conv     = nullptr;
    struct ggml_tensor * out_proj = nullptr;
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attn_norm       = nullptr;
    struct ggml_tensor * attn_norm_b     = nullptr;
    struct ggml_tensor * attn_norm_2     = nullptr;
    struct ggml_tensor * attn_norm_2_b   = nullptr;
    struct ggml_tensor * attn_q_norm     = nullptr;
    struct ggml_tensor * attn_q_norm_b   = nullptr;
    struct ggml_tensor * attn_k_norm     = nullptr;
    struct ggml_tensor * attn_k_norm_b   = nullptr;
    struct ggml_tensor * attn_out_norm   = nullptr;
    struct ggml_tensor * attn_out_norm_b = nullptr;
    struct ggml_tensor * attn_q_a_norm   = nullptr;
    struct ggml_tensor * attn_kv_a_norm  = nullptr;
    struct ggml_tensor * attn_sub_norm   = nullptr;
    struct ggml_tensor * attn_post_norm  = nullptr;
    struct ggml_tensor * ffn_sub_norm    = nullptr;
    struct ggml_tensor * attn_norm_cross = nullptr;
    struct ggml_tensor * attn_norm_enc   = nullptr;
    struct ggml_tensor * ssm_norm        = nullptr;
    struct ggml_tensor * ssm_dt_norm     = nullptr;
    struct ggml_tensor * ssm_b_norm      = nullptr;
    struct ggml_tensor * ssm_c_norm      = nullptr;

    // attention
    struct ggml_tensor * wq        = nullptr;
    struct ggml_tensor * wk        = nullptr;
    struct ggml_tensor * wv        = nullptr;
    struct ggml_tensor * wo        = nullptr;
    struct ggml_tensor * wqkv      = nullptr;
    struct ggml_tensor * wq_a      = nullptr;
    struct ggml_tensor * wq_b      = nullptr;
    struct ggml_tensor * wkv_a_mqa = nullptr;
    struct ggml_tensor * wkv_b     = nullptr;
    struct ggml_tensor * wk_b      = nullptr;
    struct ggml_tensor * wv_b      = nullptr;
    struct ggml_tensor * wq_cross  = nullptr;
    struct ggml_tensor * wk_cross  = nullptr;
    struct ggml_tensor * wv_cross  = nullptr;
    struct ggml_tensor * wo_cross  = nullptr;
    struct ggml_tensor * wq_enc    = nullptr;
    struct ggml_tensor * wk_enc    = nullptr;
    struct ggml_tensor * wv_enc    = nullptr;
    struct ggml_tensor * wo_enc    = nullptr;

    // attention bias
    struct ggml_tensor * bq   = nullptr;
    struct ggml_tensor * bk   = nullptr;
    struct ggml_tensor * bv   = nullptr;
    struct ggml_tensor * bo   = nullptr;
    struct ggml_tensor * bqkv = nullptr;

    // relative position bias
    struct ggml_tensor * attn_rel_b       = nullptr;
    struct ggml_tensor * attn_rel_b_enc   = nullptr;
    struct ggml_tensor * attn_rel_b_cross = nullptr;

    // normalization
    struct ggml_tensor * ffn_norm         = nullptr;
    struct ggml_tensor * ffn_norm_b       = nullptr;
    struct ggml_tensor * ffn_post_norm    = nullptr;
    struct ggml_tensor * layer_out_norm   = nullptr;
    struct ggml_tensor * layer_out_norm_b = nullptr;
    struct ggml_tensor * ffn_norm_exps    = nullptr;
    struct ggml_tensor * ffn_norm_enc     = nullptr;

    // ff
    struct ggml_tensor * ffn_gate     = nullptr; // w1
    struct ggml_tensor * ffn_down     = nullptr; // w2
    struct ggml_tensor * ffn_up       = nullptr; // w3
    struct ggml_tensor * ffn_gate_enc = nullptr;
    struct ggml_tensor * ffn_down_enc = nullptr;
    struct ggml_tensor * ffn_up_enc   = nullptr;

    // ff MoE
    struct ggml_tensor * ffn_gate_inp  = nullptr;
    struct ggml_tensor * ffn_gate_exps = nullptr;
    struct ggml_tensor * ffn_down_exps = nullptr;
    struct ggml_tensor * ffn_up_exps   = nullptr;

    // ff shared expert (shexp)
    struct ggml_tensor * ffn_gate_inp_shexp = nullptr;
    struct ggml_tensor * ffn_gate_shexp     = nullptr;
    struct ggml_tensor * ffn_down_shexp     = nullptr;
    struct ggml_tensor * ffn_up_shexp       = nullptr;

    // ff bias
    struct ggml_tensor * ffn_gate_b = nullptr;
    struct ggml_tensor * ffn_down_b = nullptr; // b2
    struct ggml_tensor * ffn_up_b   = nullptr; // b3
    struct ggml_tensor * ffn_act    = nullptr;
    struct ggml_tensor * ffn_exp_probs_b = nullptr;

    // mamba proj
    struct ggml_tensor * ssm_in  = nullptr;
    struct ggml_tensor * ssm_x   = nullptr;
    struct ggml_tensor * ssm_dt  = nullptr;
    struct ggml_tensor * ssm_out = nullptr;

    // mamba
    struct ggml_tensor * ssm_conv1d = nullptr;
    struct ggml_tensor * ssm_a      = nullptr;
    struct ggml_tensor * ssm_d      = nullptr;

    // mamba bias
    struct ggml_tensor * ssm_conv1d_b = nullptr;
    struct ggml_tensor * ssm_dt_b     = nullptr;

    // rwkv
    struct ggml_tensor * time_mix_w1         = nullptr;
    struct ggml_tensor * time_mix_w2         = nullptr;
    struct ggml_tensor * time_mix_lerp_x     = nullptr;
    struct ggml_tensor * time_mix_lerp_w     = nullptr;
    struct ggml_tensor * time_mix_lerp_k     = nullptr;
    struct ggml_tensor * time_mix_lerp_v     = nullptr;
    struct ggml_tensor * time_mix_lerp_r     = nullptr;
    struct ggml_tensor * time_mix_lerp_g     = nullptr;
    struct ggml_tensor * time_mix_lerp_fused = nullptr;

    struct ggml_tensor * time_mix_first        = nullptr;
    struct ggml_tensor * time_mix_decay        = nullptr;
    struct ggml_tensor * time_mix_decay_w1     = nullptr;
    struct ggml_tensor * time_mix_decay_w2     = nullptr;
    struct ggml_tensor * time_mix_key          = nullptr;
    struct ggml_tensor * time_mix_key_b        = nullptr;
    struct ggml_tensor * time_mix_value        = nullptr;
    struct ggml_tensor * time_mix_value_b      = nullptr;
    struct ggml_tensor * time_mix_receptance   = nullptr;
    struct ggml_tensor * time_mix_receptance_b = nullptr;
    struct ggml_tensor * time_mix_gate         = nullptr;

    // rwkv7
    struct ggml_tensor * time_mix_w0         = nullptr;
    struct ggml_tensor * time_mix_a0         = nullptr;
    struct ggml_tensor * time_mix_a1         = nullptr;
    struct ggml_tensor * time_mix_a2         = nullptr;
    struct ggml_tensor * time_mix_v0         = nullptr;
    struct ggml_tensor * time_mix_v1         = nullptr;
    struct ggml_tensor * time_mix_v2         = nullptr;
    struct ggml_tensor * time_mix_g1         = nullptr;
    struct ggml_tensor * time_mix_g2         = nullptr;
    struct ggml_tensor * time_mix_k_k        = nullptr;
    struct ggml_tensor * time_mix_k_a        = nullptr;
    struct ggml_tensor * time_mix_r_k        = nullptr;

    struct ggml_tensor * time_mix_ln     = nullptr;
    struct ggml_tensor * time_mix_ln_b   = nullptr;
    struct ggml_tensor * time_mix_output = nullptr;

    struct ggml_tensor * channel_mix_lerp_k = nullptr;
    struct ggml_tensor * channel_mix_lerp_r = nullptr;

    struct ggml_tensor * channel_mix_key        = nullptr;
    struct ggml_tensor * channel_mix_receptance = nullptr;
    struct ggml_tensor * channel_mix_value      = nullptr;

    // long rope factors
    struct ggml_tensor * rope_long  = nullptr;
    struct ggml_tensor * rope_short = nullptr;
    struct ggml_tensor * rope_freqs = nullptr;

    // bitnet scale
    struct ggml_tensor * wq_scale       = nullptr;
    struct ggml_tensor * wk_scale       = nullptr;
    struct ggml_tensor * wv_scale       = nullptr;
    struct ggml_tensor * wo_scale       = nullptr;
    struct ggml_tensor * ffn_gate_scale = nullptr;
    struct ggml_tensor * ffn_up_scale   = nullptr;
    struct ggml_tensor * ffn_down_scale = nullptr;

    // altup & laurel
    struct ggml_tensor * per_layer_inp_gate   = nullptr;
    struct ggml_tensor * per_layer_proj       = nullptr;
    struct ggml_tensor * per_layer_post_norm  = nullptr;
    struct ggml_tensor * altup_correct_coef   = nullptr;
    struct ggml_tensor * altup_correct_scale  = nullptr;
    struct ggml_tensor * altup_predict_coef   = nullptr;
    struct ggml_tensor * altup_router         = nullptr;
    struct ggml_tensor * altup_router_norm    = nullptr;
    struct ggml_tensor * laurel_l             = nullptr;
    struct ggml_tensor * laurel_r             = nullptr;
    struct ggml_tensor * laurel_post_norm     = nullptr;

    struct llama_layer_posnet posnet;

    struct llama_layer_convnext convnext;

    struct llama_layer_shortconv shortconv;


    //CosyVoiceFlow
    //encoders
    struct ggml_tensor * encoders_wq           = nullptr;
    struct ggml_tensor * encoders_wk           = nullptr;
    struct ggml_tensor * encoders_wv           = nullptr;
    struct ggml_tensor * encoders_wo           = nullptr;
    struct ggml_tensor * encoders_wpos         = nullptr;
    
    struct ggml_tensor * encoders_bq           = nullptr;
    struct ggml_tensor * encoders_bk           = nullptr;
    struct ggml_tensor * encoders_bv           = nullptr;
    struct ggml_tensor * encoders_bo           = nullptr;
    
    struct ggml_tensor * encoders_ffn_w1       = nullptr;
    struct ggml_tensor * encoders_ffn_w2       = nullptr;
    struct ggml_tensor * encoders_ffn_b1       = nullptr;
    struct ggml_tensor * encoders_ffn_b2       = nullptr;

    struct ggml_tensor * encoders_normffn_w    = nullptr;
    struct ggml_tensor * encoders_normmha_w    = nullptr;
    struct ggml_tensor * encoders_normffn_b    = nullptr;
    struct ggml_tensor * encoders_normmha_b    = nullptr;
    struct ggml_tensor * encoders_pos_bias_u   = nullptr;
    struct ggml_tensor * encoders_pos_bias_v   = nullptr;

    //up_encoders
    struct ggml_tensors * up_encoders_wq        = nullptr;
    struct ggml_tensors * up_encoders_wk        = nullptr;
    struct ggml_tensors * up_encoders_wv        = nullptr;
    struct ggml_tensors * up_encoders_wo        = nullptr;
    struct ggml_tensors * up_encoders_wpos      = nullptr;
    struct ggml_tensors * up_encoders_bq        = nullptr;
    struct ggml_tensors * up_encoders_bk        = nullptr;
    struct ggml_tensors * up_encoders_bv        = nullptr;
    struct ggml_tensors * up_encoders_bo        = nullptr;
    struct ggml_tensors * up_encoders_ffn_w1    = nullptr;
    struct ggml_tensors * up_encoders_ffn_w2    = nullptr;
    struct ggml_tensors * up_encoders_ffn_b1    = nullptr;
    struct ggml_tensors * up_encoders_ffn_b2    = nullptr;
    struct ggml_tensors * up_encoders_normffn_w = nullptr;
    struct ggml_tensors * up_encoders_normmha_w = nullptr;
    struct ggml_tensor *  up_encoders_normffn_b = nullptr;
    struct ggml_tensor *  up_encoders_normmha_b = nullptr;
    struct ggml_tensor *  up_encoders_pos_bias_u= nullptr;
    struct ggml_tensor *  up_encoders_pos_bias_v= nullptr;

    //decoder
    // down_block
    struct ggml_tensor * down_block1_norm1_w           = nullptr;
    struct ggml_tensor * down_block1_norm1_b           = nullptr;
    struct ggml_tensor * down_block1_norm3_w           = nullptr;
    struct ggml_tensor * down_block1_norm3_b           = nullptr;
    struct ggml_tensor * down_block1_wq                = nullptr;
    struct ggml_tensor * down_block1_wk                = nullptr;
    struct ggml_tensor * down_block1_wv                = nullptr;
    struct ggml_tensor * down_block1_wo                = nullptr;
    struct ggml_tensor * down_block1_bo                = nullptr;
    struct ggml_tensor * down_block1_ffn_w0            = nullptr;
    struct ggml_tensor * down_block1_ffn_w2            = nullptr;
    struct ggml_tensor * down_block1_ffn_b0            = nullptr;
    struct ggml_tensor * down_block1_ffn_b2            = nullptr;

    //mid_block
    struct ggml_tensor * mid_block_mlp_w               = nullptr;
    struct ggml_tensor * mid_block_mlp_b               = nullptr;
    struct ggml_tensor * mid_block1_w                  = nullptr;
    struct ggml_tensor * mid_block1_b                  = nullptr;
    struct ggml_tensor * mid_block1_norm_w             = nullptr;
    struct ggml_tensor * mid_block1_norm_b             = nullptr;                  
    struct ggml_tensor * mid_block2_w                  = nullptr;
    struct ggml_tensor * mid_block2_b                  = nullptr;
    struct ggml_tensor * mid_block2_norm_w             = nullptr;
    struct ggml_tensor * mid_block2_norm_b             = nullptr;
    struct ggml_tensor * mid_block_res_w               = nullptr;
    struct ggml_tensor * mid_block_res_b               = nullptr;
    
    struct ggml_tensor * mid_block1_norm1_w           = nullptr;
    struct ggml_tensor * mid_block1_norm1_b           = nullptr;
    struct ggml_tensor * mid_block1_norm3_w           = nullptr;
    struct ggml_tensor * mid_block1_norm3_b           = nullptr;
    struct ggml_tensor * mid_block1_wq                = nullptr;
    struct ggml_tensor * mid_block1_wk                = nullptr;
    struct ggml_tensor * mid_block1_wv                = nullptr;
    struct ggml_tensor * mid_block1_wo                = nullptr;
    struct ggml_tensor * mid_block1_bo                = nullptr;
    struct ggml_tensor * mid_block1_ffn_w0            = nullptr;
    struct ggml_tensor * mid_block1_ffn_w2            = nullptr;
    struct ggml_tensor * mid_block1_ffn_b0            = nullptr;
    struct ggml_tensor * mid_block1_ffn_b2            = nullptr;

    //up_block
    struct ggml_tensor * up_block1_norm1_w           = nullptr;
    struct ggml_tensor * up_block1_norm1_b           = nullptr;
    struct ggml_tensor * up_block1_norm3_w           = nullptr;
    struct ggml_tensor * up_block1_norm3_b           = nullptr;
    struct ggml_tensor * up_block1_wq                = nullptr;
    struct ggml_tensor * up_block1_wk                = nullptr;
    struct ggml_tensor * up_block1_wv                = nullptr;
    struct ggml_tensor * up_block1_wo                = nullptr;
    struct ggml_tensor * up_block1_bo                = nullptr;
    struct ggml_tensor * up_block1_ffn_w0            = nullptr;
    struct ggml_tensor * up_block1_ffn_w2            = nullptr;
    struct ggml_tensor * up_block1_ffn_b0            = nullptr;
    struct ggml_tensor * up_block1_ffn_b2            = nullptr; 
};

struct llama_model {
    llm_type type = LLM_TYPE_UNKNOWN;
    llm_arch arch = LLM_ARCH_UNKNOWN;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    // for classifier models
    std::vector<std::string> classifier_labels;

    struct ggml_tensor * tok_embd   = nullptr;
    struct ggml_tensor * type_embd  = nullptr;
    struct ggml_tensor * pos_embd   = nullptr;
    struct ggml_tensor * tok_norm   = nullptr;
    struct ggml_tensor * tok_norm_b = nullptr;

    struct ggml_tensor * output_norm     = nullptr;
    struct ggml_tensor * output_norm_b   = nullptr;
    struct ggml_tensor * output          = nullptr;
    struct ggml_tensor * output_b        = nullptr;
    struct ggml_tensor * output_norm_enc = nullptr;

    // classifier
    struct ggml_tensor * cls       = nullptr;
    struct ggml_tensor * cls_b     = nullptr;
    struct ggml_tensor * cls_out   = nullptr;
    struct ggml_tensor * cls_out_b = nullptr;

    struct ggml_tensor * conv1d   = nullptr;
    struct ggml_tensor * conv1d_b = nullptr;

    // gemma3n altup
    struct ggml_tensor * tok_embd_per_layer   = nullptr;
    struct ggml_tensor * altup_proj           = nullptr;
    struct ggml_tensor * altup_unembd_proj    = nullptr;
    struct ggml_tensor * per_layer_model_proj = nullptr;
    struct ggml_tensor * per_layer_proj_norm  = nullptr;

    // CosyVoiceFlow
    struct ggml_tensor * decoder_estimator_down_blocks_0_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_final_block_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_final_block_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_final_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_time_mlp_linear_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_time_mlp_linear_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_0_block1_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_0_block1_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_0_block2_block_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_0_block2_block_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_0_mlp_1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_0_res_conv_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_attn1_to_k_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_attn1_to_out_0_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_attn1_to_q_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_attn1_to_v_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_ff_net_0_proj_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_ff_net_2_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_norm1_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_norm3_weight = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_2_weight = nullptr;
    struct ggml_tensor * encoder_after_norm_weight = nullptr;
    struct ggml_tensor * encoder_embed_out_0_weight = nullptr;
    struct ggml_tensor * encoder_embed_out_1_weight = nullptr;
    struct ggml_tensor * encoder_pre_lookahead_layer_conv1_weight = nullptr;
    struct ggml_tensor * encoder_pre_lookahead_layer_conv2_weight = nullptr;
    struct ggml_tensor * encoder_up_embed_out_0_weight = nullptr;
    struct ggml_tensor * encoder_up_embed_out_1_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_feed_forward_w_1_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_feed_forward_w_2_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_norm_ff_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_norm_mha_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_self_attn_linear_k_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_self_attn_linear_out_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_self_attn_linear_pos_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_self_attn_linear_q_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_self_attn_linear_v_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_feed_forward_w_1_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_feed_forward_w_2_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_norm_ff_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_norm_mha_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_self_attn_linear_k_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_self_attn_linear_out_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_self_attn_linear_pos_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_self_attn_linear_q_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_self_attn_linear_v_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_feed_forward_w_1_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_feed_forward_w_2_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_norm_ff_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_norm_mha_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_self_attn_linear_k_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_self_attn_linear_out_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_self_attn_linear_pos_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_self_attn_linear_q_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_self_attn_linear_v_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_feed_forward_w_1_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_feed_forward_w_2_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_norm_ff_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_norm_mha_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_self_attn_linear_k_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_self_attn_linear_out_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_self_attn_linear_pos_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_self_attn_linear_q_weight = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_self_attn_linear_v_weight = nullptr;
    struct ggml_tensor * encoder_up_layer_conv_weight = nullptr;
    struct ggml_tensor * encoder_proj_weight = nullptr;
    struct ggml_tensor * input_embedding_weight = nullptr;
    struct ggml_tensor * spk_embed_affine_layer_weight = nullptr;

    struct ggml_tensor * decoder_estimator_down_blocks_0_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_down_blocks_0_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_final_block_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_final_block_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_final_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_0_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_1_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_10_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_11_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_2_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_3_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_4_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_5_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_6_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_7_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_8_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_mid_blocks_9_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_time_mlp_linear_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_time_mlp_linear_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_0_block1_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_0_block1_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_0_block2_block_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_0_block2_block_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_0_mlp_1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_0_res_conv_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_0_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_1_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_2_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_attn1_to_out_0_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_ff_net_0_proj_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_ff_net_2_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_norm1_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_1_3_norm3_bias = nullptr;
    struct ggml_tensor * decoder_estimator_up_blocks_0_2_bias = nullptr;
    struct ggml_tensor * encoder_after_norm_bias = nullptr;
    struct ggml_tensor * encoder_embed_out_0_bias = nullptr;
    struct ggml_tensor * encoder_embed_out_1_bias = nullptr;
    struct ggml_tensor * encoder_encoders_0_feed_forward_w_1_bias = nullptr;
    struct ggml_tensor * encoder_encoders_0_feed_forward_w_2_bias = nullptr;
    struct ggml_tensor * encoder_encoders_0_norm_ff_bias = nullptr;
    struct ggml_tensor * encoder_encoders_0_norm_mha_bias = nullptr;
    struct ggml_tensor * encoder_encoders_0_self_attn_linear_k_bias = nullptr;
    struct ggml_tensor * encoder_encoders_0_self_attn_linear_out_bias = nullptr;
    struct ggml_tensor * encoder_encoders_0_self_attn_linear_q_bias = nullptr;
    struct ggml_tensor * encoder_encoders_0_self_attn_linear_v_bias = nullptr;
    struct ggml_tensor * encoder_encoders_0_self_attn_pos_bias_u = nullptr;
    struct ggml_tensor * encoder_encoders_0_self_attn_pos_bias_v = nullptr;
    struct ggml_tensor * encoder_encoders_1_feed_forward_w_1_bias = nullptr;
    struct ggml_tensor * encoder_encoders_1_feed_forward_w_2_bias = nullptr;
    struct ggml_tensor * encoder_encoders_1_norm_ff_bias = nullptr;
    struct ggml_tensor * encoder_encoders_1_norm_mha_bias = nullptr;
    struct ggml_tensor * encoder_encoders_1_self_attn_linear_k_bias = nullptr;
    struct ggml_tensor * encoder_encoders_1_self_attn_linear_out_bias = nullptr;
    struct ggml_tensor * encoder_encoders_1_self_attn_linear_q_bias = nullptr;
    struct ggml_tensor * encoder_encoders_1_self_attn_linear_v_bias = nullptr;
    struct ggml_tensor * encoder_encoders_1_self_attn_pos_bias_u = nullptr;
    struct ggml_tensor * encoder_encoders_1_self_attn_pos_bias_v = nullptr;
    struct ggml_tensor * encoder_encoders_2_feed_forward_w_1_bias = nullptr;
    struct ggml_tensor * encoder_encoders_2_feed_forward_w_2_bias = nullptr;
    struct ggml_tensor * encoder_encoders_2_norm_ff_bias = nullptr;
    struct ggml_tensor * encoder_encoders_2_norm_mha_bias = nullptr;
    struct ggml_tensor * encoder_encoders_2_self_attn_linear_k_bias = nullptr;
    struct ggml_tensor * encoder_encoders_2_self_attn_linear_out_bias = nullptr;
    struct ggml_tensor * encoder_encoders_2_self_attn_linear_q_bias = nullptr;
    struct ggml_tensor * encoder_encoders_2_self_attn_linear_v_bias = nullptr;
    struct ggml_tensor * encoder_encoders_2_self_attn_pos_bias_u = nullptr;
    struct ggml_tensor * encoder_encoders_2_self_attn_pos_bias_v = nullptr;
    struct ggml_tensor * encoder_encoders_3_feed_forward_w_1_bias = nullptr;
    struct ggml_tensor * encoder_encoders_3_feed_forward_w_2_bias = nullptr;
    struct ggml_tensor * encoder_encoders_3_norm_ff_bias = nullptr;
    struct ggml_tensor * encoder_encoders_3_norm_mha_bias = nullptr;
    struct ggml_tensor * encoder_encoders_3_self_attn_linear_k_bias = nullptr;
    struct ggml_tensor * encoder_encoders_3_self_attn_linear_out_bias = nullptr;
    struct ggml_tensor * encoder_encoders_3_self_attn_linear_q_bias = nullptr;
    struct ggml_tensor * encoder_encoders_3_self_attn_linear_v_bias = nullptr;
    struct ggml_tensor * encoder_encoders_3_self_attn_pos_bias_u = nullptr;
    struct ggml_tensor * encoder_encoders_3_self_attn_pos_bias_v = nullptr;
    struct ggml_tensor * encoder_encoders_4_feed_forward_w_1_bias = nullptr;
    struct ggml_tensor * encoder_encoders_4_feed_forward_w_2_bias = nullptr;
    struct ggml_tensor * encoder_encoders_4_norm_ff_bias = nullptr;
    struct ggml_tensor * encoder_encoders_4_norm_mha_bias = nullptr;
    struct ggml_tensor * encoder_encoders_4_self_attn_linear_k_bias = nullptr;
    struct ggml_tensor * encoder_encoders_4_self_attn_linear_out_bias = nullptr;
    struct ggml_tensor * encoder_encoders_4_self_attn_linear_q_bias = nullptr;
    struct ggml_tensor * encoder_encoders_4_self_attn_linear_v_bias = nullptr;
    struct ggml_tensor * encoder_encoders_4_self_attn_pos_bias_u = nullptr;
    struct ggml_tensor * encoder_encoders_4_self_attn_pos_bias_v = nullptr;
    struct ggml_tensor * encoder_encoders_5_feed_forward_w_1_bias = nullptr;
    struct ggml_tensor * encoder_encoders_5_feed_forward_w_2_bias = nullptr;
    struct ggml_tensor * encoder_encoders_5_norm_ff_bias = nullptr;
    struct ggml_tensor * encoder_encoders_5_norm_mha_bias = nullptr;
    struct ggml_tensor * encoder_encoders_5_self_attn_linear_k_bias = nullptr;
    struct ggml_tensor * encoder_encoders_5_self_attn_linear_out_bias = nullptr;
    struct ggml_tensor * encoder_encoders_5_self_attn_linear_q_bias = nullptr;
    struct ggml_tensor * encoder_encoders_5_self_attn_linear_v_bias = nullptr;
    struct ggml_tensor * encoder_encoders_5_self_attn_pos_bias_u = nullptr;
    struct ggml_tensor * encoder_encoders_5_self_attn_pos_bias_v = nullptr;
    struct ggml_tensor * encoder_pre_lookahead_layer_conv1_bias = nullptr;
    struct ggml_tensor * encoder_pre_lookahead_layer_conv2_bias = nullptr;
    struct ggml_tensor * encoder_up_embed_out_0_bias = nullptr;
    struct ggml_tensor * encoder_up_embed_out_1_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_feed_forward_w_1_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_feed_forward_w_2_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_norm_ff_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_norm_mha_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_self_attn_linear_k_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_self_attn_linear_out_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_self_attn_linear_q_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_self_attn_linear_v_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_self_attn_pos_bias_u = nullptr;
    struct ggml_tensor * encoder_up_encoders_0_self_attn_pos_bias_v = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_feed_forward_w_1_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_feed_forward_w_2_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_norm_ff_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_norm_mha_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_self_attn_linear_k_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_self_attn_linear_out_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_self_attn_linear_q_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_self_attn_linear_v_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_self_attn_pos_bias_u = nullptr;
    struct ggml_tensor * encoder_up_encoders_1_self_attn_pos_bias_v = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_feed_forward_w_1_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_feed_forward_w_2_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_norm_ff_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_norm_mha_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_self_attn_linear_k_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_self_attn_linear_out_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_self_attn_linear_q_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_self_attn_linear_v_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_self_attn_pos_bias_u = nullptr;
    struct ggml_tensor * encoder_up_encoders_2_self_attn_pos_bias_v = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_feed_forward_w_1_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_feed_forward_w_2_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_norm_ff_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_norm_mha_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_self_attn_linear_k_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_self_attn_linear_out_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_self_attn_linear_q_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_self_attn_linear_v_bias = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_self_attn_pos_bias_u = nullptr;
    struct ggml_tensor * encoder_up_encoders_3_self_attn_pos_bias_v = nullptr;
    struct ggml_tensor * encoder_up_layer_conv_bias = nullptr;
    struct ggml_tensor * encoder_proj_bias = nullptr;
    struct ggml_tensor * spk_embed_affine_layer_bias = nullptr;

    


    std::vector<llama_layer> layers;

    llama_model_params params;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // list of devices used in this model
    std::vector<ggml_backend_dev_t> devices;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    explicit llama_model(const struct llama_model_params & params);
    ~llama_model();

    void load_stats  (llama_model_loader & ml);
    void load_arch   (llama_model_loader & ml);
    void load_hparams(llama_model_loader & ml);
    void load_vocab  (llama_model_loader & ml);
    bool load_tensors(llama_model_loader & ml); // returns false if cancelled by progress_callback

    std::string arch_name() const;
    std::string type_name() const;

    std::string desc() const;

    size_t size() const;
    size_t n_tensors() const;
    size_t n_devices() const;

    // total number of parameters in the model
    uint64_t n_elements() const;

    void print_info() const;

    ggml_backend_dev_t dev_layer(int il) const;
    ggml_backend_dev_t dev_output() const;

    ggml_backend_buffer_type_t select_buft(int il) const;

    bool has_tensor_overrides() const;

    const struct ggml_tensor * get_tensor(const char * name) const;

    float get_rope_freq_base (const llama_cparams & cparams, int il) const;
    float get_rope_freq_scale(const llama_cparams & cparams, int il) const;

    ggml_tensor * get_rope_factors(const llama_cparams & cparams, int il) const;

    // note: can mutate `cparams`
    // TODO: move this to new llm_arch_model_i interface
    llama_memory_i * create_memory(const llama_memory_params & params, llama_cparams & cparams) const;

    // TODO: move this to new llm_arch_model_i interface
    llm_graph_result_ptr build_graph(
            const llm_graph_params & params,
                       ggml_cgraph * gf,
                    llm_graph_type   type) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

const char * llm_type_name(llm_type type);

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);


