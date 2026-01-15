
#include "models.h"

llm_build_flow::llm_build_flow(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {

    ggml_tensor * embedding = build_inp_embd(model.inp_embed_w); //✅
    ggml_build_forward_expand(gf, embedding);
    
    ggml_tensor * token = build_inp_token();            //✅
    ggml_build_forward_expand(gf, token);

    ggml_tensor * prompt_feat = build_inp_prompt_feat(); //✅
    ggml_build_forward_expand(gf, prompt_feat);

    // ggml_tensor * test = ggml_add(ctx0, prompt_feat, prompt_feat);
    // ggml_build_forward_expand(gf, test);
    // cb(test, "test_after_attn", 0);
    
    ggml_tensor * extend_pe = build_inp_extend_pe();    //✅
    ggml_build_forward_expand(gf, extend_pe);

    embedding = build_F_normalize(embedding, 1e-12f); //✅
    
    ggml_tensor * spk_mul = ggml_mul_mat(ctx0, model.spk_embed_w, embedding); 
    ggml_tensor * spk_add = ggml_add(ctx0, spk_mul, model.spk_embed_b);       //✅
    ggml_set_name(spk_add, "spk_affine_layer");
    ggml_build_forward_expand(gf, spk_add);

    // ggml_tensor * mask = build_pad_mask(params.ubatch.prompt_token_len + params.ubatch.token_len, 0, 0);
    // ggml_build_forward_expand(gf, mask);
    // mask = ggml_cont(ctx0, ggml_permute(ctx0, mask, 1, 0, 2, 3));   //✅

    token = build_flow_embedding(token, model.inp_embed_w, -1);   //✅
    // ggml_tensor * token_mask = ggml_mul(ctx0, token, mask);
    // ggml_set_name(token_mask, "flow_embd_token");                 //✅

    // encoder
    // const int B  = token_mask->ne[2];
    // const int T  = token_mask->ne[1];
    // ggml_tensor * masks = build_pad_mask(params.ubatch.prompt_token_len + params.ubatch.token_len, T, 1);
    // masks = ggml_cont(ctx0, masks);
    // masks = ggml_reshape_3d(ctx0, masks, B, 1, T);
    
    ggml_tensor * x = build_linear_no_subsampling(token, model.embed_out_0_w, model.embed_out_0_b, model.embed_out_1_w, model.embed_out_1_b, 1); //✅
    
    x = build_espnet_pos_encode(x, 1);      //✅

    ggml_tensor * extend_pe_cpy = ggml_dup(ctx0, extend_pe);
    ggml_set_name(extend_pe_cpy, "extend_pe_cpy");
    ggml_tensor * pos_emb = build_pos_encoding(extend_pe_cpy, x->ne[1], 0, 1); //✅
    // ggml_tensor * mask_pad = masks;
    // ggml_tensor * chunk_mask = masks;
    // ggml_tensor * pre_look_mw = flip_weight(gf, model.pre_look_conv1_w);
    x = build_pre_lookahead_layer(x, model.pre_look_conv1_w, model.pre_look_conv1_b, model.pre_look_conv2_w, model.pre_look_conv2_b);  //✅
    ggml_build_forward_expand(gf, x);

    //encoders
    for(int i = 0; i < 6; i++) {
        ggml_tensor * attn_residual = ggml_dup(ctx0, x);
        ggml_build_forward_expand(gf, attn_residual);
        cb(attn_residual, "before_x_residual", i);
        x = build_layer_norm(x, model.layers[i + 13].encoders_normmha_w, model.layers[i + 13].encoders_normmha_b, 1e-12, "encoders", i);
        ggml_tensor * query = build_rel_pos_attn(gf, x, model.layers[i + 13].encoders_wq, model.layers[i + 13].encoders_bq);
        cb(query, "encoder_attn_q", i);
        ggml_tensor * key = build_rel_pos_attn(gf, x, model.layers[i + 13].encoders_wk, model.layers[i + 13].encoders_bk);
        cb(key, "encoder_attn_k", i);
        ggml_tensor * value = build_rel_pos_attn(gf, x, model.layers[i + 13].encoders_wv, model.layers[i + 13].encoders_bv);
        cb(value, "encoder_attn_v", i);
        query = ggml_cont(ctx0, ggml_permute(ctx0, query, 0, 2, 1, 3));
        int n_batch_pos = pos_emb->ne[2];
        ggml_tensor * p = ggml_mul_mat(ctx0, model.layers[i + 13].encoders_wpos, pos_emb);
        p = ggml_cont(ctx0, p);
        p = ggml_reshape_4d(ctx0, p, 64, 8, p->ne[1], n_batch_pos);
        p = ggml_cont(ctx0, ggml_permute(ctx0, p, 0, 2, 1, 3));
        ggml_tensor * q_with_bias_u = ggml_cont(ctx0, ggml_permute(ctx0, ggml_add(ctx0, query, model.layers[i + 13].encoders_pos_bias_u), 0, 2, 1, 3));
        cb(q_with_bias_u, "q_with_bias_u", i);
        ggml_tensor * q_with_bias_v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_add(ctx0, query, model.layers[i + 13].encoders_pos_bias_v), 0, 2, 1, 3));
        cb(q_with_bias_v, "q_with_bias_v", i);
        ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, key, q_with_bias_u);
        cb(matrix_ac, "matrix_ac", i);
        ggml_tensor * matrix_bd = ggml_mul_mat(ctx0, p, q_with_bias_v);
        cb(matrix_bd, "matrix_bd", i);
        matrix_bd = build_rel_shift(gf, matrix_bd);
        cb(matrix_bd, "matrix_bd_rel_shift", i);
        ggml_tensor * ac_bd = ggml_add(ctx0, matrix_ac, matrix_bd);
        cb(ac_bd, "ac_plus_bd", i);
        ggml_tensor * scores = ggml_scale(ctx0, ac_bd, 1.0f / sqrtf(float(64.0f)));
        cb(scores, "scores", i);
        ggml_tensor * x_att = build_attn_scores(value, scores, model.layers[i + 13].encoders_wo, model.layers[i + 13].encoders_bo, "encoders", i);
        cb(x_att, "x_att", i);
        cb(attn_residual, "residual", i);
        x = ggml_add(ctx0, attn_residual, x_att);
        cb(x, "res+x", i);
        ggml_tensor * ffn_residual = ggml_dup(ctx0, x);
        ggml_build_forward_expand(gf, ffn_residual);
        x = build_layer_norm(x, model.layers[i + 13].encoders_normffn_w, model.layers[i + 13].encoders_normffn_b, 1e-12, "encoders", i * 6 + 1);
        cb(x, "normffn", i);
        x = build_pos_ffn(x, model.layers[i + 13].encoders_ffn_w1, model.layers[i + 13].encoders_ffn_b1, model.layers[i + 13].encoders_ffn_w2, model.layers[i + 13].encoders_ffn_b2);
        cb(x, "ffn_out", i);
        x = ggml_add(ctx0, ffn_residual, x);
        cb(x, "encoders_out", i);
    }
    ggml_set_name(x, "after encoders");
    x = build_upsample_1d(gf, x, model.up_layer_conv_w, model.up_layer_conv_b);
    ggml_set_name(x, "after build_upsample_1d");
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    // masks = build_pad_mask(x->ne[1], x->ne[1], 2);
    // masks = ggml_cont(ctx0, masks);
    // masks = ggml_reshape_3d(ctx0, masks, masks->ne[0], 1, masks->ne[1]);
    x = build_linear_no_subsampling(x, model.up_embed_out_0_w, model.up_embed_out_0_b, model.up_embed_out_1_w, model.up_embed_out_1_b, 2);
    x = build_espnet_pos_encode(x, 2);
    ggml_tensor * extend_pe_cpy2 = ggml_dup(ctx0, extend_pe);
    pos_emb = build_pos_encoding(extend_pe_cpy2, x->ne[1], 0, 2);
    // mask_pad = masks;
    // chunk_mask = masks;
    // x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));

    //build up_encoders
    for(int i = 0; i < 4; i++) {
        ggml_tensor * attn_residual = ggml_dup(ctx0, x);
        ggml_build_forward_expand(gf, attn_residual);
        cb(attn_residual, "up_encoders_x_residual", i);
        x = build_layer_norm(x, model.layers[i + 132].up_encoders_normmha_w, model.layers[i + 132].up_encoders_normmha_b, 1e-12, "up_encoders", 12 + i);
        // x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
        ggml_tensor * query = build_rel_pos_attn(gf, x, model.layers[i + 132].up_encoders_wq, model.layers[i + 132].up_encoders_bq);
        cb(query, "up_encoders_attn_q", i);
        ggml_tensor * key = build_rel_pos_attn(gf, x, model.layers[i + 132].up_encoders_wk, model.layers[i + 132].up_encoders_bk);
        cb(key, "up_encoders_attn_k", i);
        ggml_tensor * value = build_rel_pos_attn(gf, x, model.layers[i + 132].up_encoders_wv, model.layers[i + 132].up_encoders_bv);
        cb(value, "up_encoders_attn_v", i);
        query = ggml_cont(ctx0, ggml_permute(ctx0, query, 0, 2, 1, 3));
        int n_batch_pos = pos_emb->ne[2];
        ggml_tensor * p = ggml_mul_mat(ctx0, model.layers[i + 132].up_encoders_wpos, pos_emb);
        p = ggml_cont(ctx0, p);
        p = ggml_reshape_4d(ctx0, p, 64, 8, p->ne[1], n_batch_pos);
        p = ggml_cont(ctx0, ggml_permute(ctx0, p, 0, 2, 1, 3));
        ggml_tensor * q_with_bias_u = ggml_cont(ctx0, ggml_permute(ctx0, ggml_add(ctx0, query, model.layers[i + 132].up_encoders_pos_bias_u), 0, 2, 1, 3));
        cb(q_with_bias_u, "up_encoder_q_with_bias_u", i);
        ggml_tensor * q_with_bias_v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_add(ctx0, query, model.layers[i + 132].up_encoders_pos_bias_v), 0, 2, 1, 3));
        cb(q_with_bias_v, "up_encoder_q_with_bias_v", i);
        ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, key, q_with_bias_u);
        ggml_tensor * matrix_bd = ggml_mul_mat(ctx0, p, q_with_bias_v);
        matrix_bd = build_rel_shift(gf, matrix_bd);
        ggml_tensor * scores = ggml_scale(ctx0, ggml_add(ctx0, matrix_ac, matrix_bd), 1.0f / sqrtf(float(64.0f)));
        cb(scores, "up_scores", i);
        ggml_tensor * x_att = build_attn_scores(value, scores, model.layers[i + 132].up_encoders_wo, model.layers[i + 132].up_encoders_bo, "up_encoders", i);
        cb(x_att, "up_x_att", i);
        x = ggml_add(ctx0, attn_residual, x_att);
        ggml_tensor * ffn_residual = ggml_dup(ctx0, x);
        ggml_build_forward_expand(gf, ffn_residual);
        x = build_layer_norm(x, model.layers[i + 132].up_encoders_normffn_w, model.layers[i + 132].up_encoders_normffn_b, 1e-12, "up_encoders", 16 + i);
        x = build_pos_ffn(x, model.layers[i + 132].up_encoders_ffn_w1, model.layers[i + 132].up_encoders_ffn_b1, model.layers[i + 132].up_encoders_ffn_w2, model.layers[i + 132].up_encoders_ffn_b2);
        cb(x, "up_fn_out", i);
        x = ggml_add(ctx0, ffn_residual, x);
        cb(x, "up_encoder_out", i);
    }
    ggml_set_name(x, "after up_encoders");
    x = build_layer_norm(x, model.after_norm_w, model.after_norm_b, 1e-5f, "before_decoder", 20);
    ggml_set_name(x, "after encoder");
    x = ggml_mul_mat(ctx0, model.encoder_proj_w, x);
    x = ggml_add(ctx0, x, model.encoder_proj_b);
    ggml_set_name(x, "after encoder_proj");

    //build decoder
    int32_t mel_len1 = prompt_feat->ne[1];
    int32_t mel_len2 = x->ne[1] - mel_len1;

    ggml_tensor * conds = ggml_new_tensor_3d(ctx0, x->type, prompt_feat->ne[0], mel_len1 + mel_len2, 1);
    conds = ggml_scale(ctx0, conds, 0.0f);
    ggml_tensor * dest_view = ggml_view_3d(ctx0, conds, prompt_feat->ne[0], mel_len1, prompt_feat->ne[2], conds->nb[1], conds->ne[2], 0);
    // ggml_cpy(ctx0, prompt_feat, dest_view);
    ggml_tensor * cpy = ggml_cpy(ctx0, prompt_feat, dest_view);
    ggml_set_name(cpy, "get_dest_view");
    ggml_build_forward_expand(gf, cpy);
    conds = ggml_cont(ctx0, ggml_transpose(ctx0, conds));
    ggml_set_name(conds, "conds");

    ggml_tensor * mask = build_pad_mask(mel_len1 + mel_len2, 0, 3);
    mask = ggml_cont(ctx0, mask);
    ggml_tensor * spks = ggml_dup(ctx0, spk_add);
    ggml_tensor * cond = ggml_dup(ctx0, conds);
    int64_t n_timesteps = 10;
    ggml_tensor * mu = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    ggml_set_name(mu, "mu");
    mask = ggml_reshape_3d(ctx0, mask, mask->ne[0], 1, mask->ne[1]);
    ggml_tensor * rand_noise = build_inp_rand_noise();
    ggml_tensor * z = ggml_view_3d(ctx0, rand_noise, mu->ne[0], rand_noise->ne[1], rand_noise->ne[2], rand_noise->nb[1], rand_noise->nb[2], 0);
    z = ggml_cont(ctx0, z);
    ggml_set_name(z, "z");
    ggml_tensor * feat = build_solve_euler(gf, z, mu, mask, spks, cond, model);
    // ggml_tensor * sliced = ggml_view_3d(ctx0, feat, feat->ne[0] - mel_len1, feat->ne[1], feat->ne[2], feat->nb[0], feat->nb[1], mel_len1 * feat->nb[0]);
    // LLAMA_LOG_INFO("&&&&&&&&&&&&&&& feat shape is: {%d, %d, %d, %d}\n", feat->ne[0], feat->ne[1], feat->ne[2], feat->ne[3]);
    cb(feat, "result_norm", -1);
    res->t_embd = feat;
    ggml_build_forward_expand(gf, feat);
    // ggml_graph_dump_dot(gf, NULL, "debug.dot");
}

