
#include "models.h"

llm_build_flow::llm_build_flow(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int32_t stream_val = build_stream();

    ggml_tensor * embedding = build_inp_embd(model.inp_embed_w); //✅
    
    ggml_tensor * token = build_inp_token();            //✅

    ggml_tensor * prompt_feat = build_inp_prompt_feat(); //✅
    
    ggml_tensor * extend_pe = build_inp_extend_pe();    //✅

    embedding = build_F_normalize(embedding, 1e-12f); //✅
    
    // ggml_tensor * spk_mul = ggml_mul_mat(ctx0, model.spk_embed_w, embedding); 
    // ggml_tensor * spk_add = ggml_add(ctx0, spk_mul, model.spk_embed_b);       //✅
    ggml_tensor * spk_add = ggml_mul_mat(ctx0, model.spk_embed_w, embedding, model.spk_embed_b);
    ggml_set_name(spk_add, "spk_affine_layer");

    token = build_flow_embedding(token, model.inp_embed_w, -1);   //✅
    
    bool stream = (stream_val != 0);
    bool finalize = !stream;
    ggml_tensor * x = nullptr;
    int32_t token_len = token->ne[1];
    if (finalize) {
        x = build_encoder(token, token_len, nullptr, extend_pe, stream, model);
    }else {
        ggml_tensor * token_stream = ggml_view_3d(
            ctx0, token, token->ne[0], token->ne[1] -3, token->ne[2], token->nb[1], token->nb[2], 0);
        ggml_tensor * context = ggml_view_3d(
            ctx0, token, token->ne[0], 3, token->ne[2], token->nb[1], token->nb[2], 0);
        x = build_encoder(token, token_len, context, extend_pe, stream, model);
    }
    
    x = build_layer_norm(x, model.after_norm_w, model.after_norm_b, 1e-5f, "before_decoder", 20);
    // x = ggml_mul_mat(ctx0, model.encoder_proj_w, x);
    // x = ggml_add(ctx0, x, model.encoder_proj_b);
    x = ggml_mul_mat_add(ctx0, model.encoder_proj_w, x, model.encoder_proj_b);

    //build decoder
    int32_t mel_len1 = prompt_feat->ne[1];
    int32_t mel_len2 = x->ne[1] - mel_len1;

    ggml_tensor * conds = ggml_new_tensor_3d(ctx0, x->type, prompt_feat->ne[0], mel_len1 + mel_len2, 1);
    conds = ggml_scale(ctx0, conds, 0.0f);
    ggml_tensor * dest_view = ggml_view_3d(ctx0, conds, prompt_feat->ne[0], mel_len1, prompt_feat->ne[2], conds->nb[1], conds->ne[2], 0);
    ggml_tensor * cpy = ggml_cpy(ctx0, prompt_feat, dest_view);
    ggml_build_forward_expand(gf, cpy);
    conds = ggml_cont(ctx0, ggml_transpose(ctx0, conds));


    ggml_tensor * mask = build_pad_mask(mel_len1 + mel_len2, 0, 3);
    mask = ggml_cont(ctx0, mask);
    ggml_tensor * spks = spk_add;
    ggml_tensor * cond = conds;
    int64_t n_timesteps = 10;
    ggml_tensor * mu = ggml_cont(ctx0, ggml_transpose(ctx0, x));

    mask = ggml_reshape_3d(ctx0, mask, mask->ne[0], 1, mask->ne[1]);
    ggml_tensor * rand_noise = build_inp_rand_noise();
    ggml_tensor * z = ggml_view_3d(ctx0, rand_noise, mu->ne[0], rand_noise->ne[1], rand_noise->ne[2], rand_noise->nb[1], rand_noise->nb[2], 0);
    z = ggml_cont(ctx0, z);

    ggml_tensor * feat = build_solve_euler(gf, z, mu, mask, spks, cond, model, stream);

    res->t_embd = feat;
    ggml_build_forward_expand(gf, feat);
}
