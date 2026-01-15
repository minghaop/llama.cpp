#include "models.h"


// 硬编码的 Hann Window (16点)
static constexpr float HANN_WINDOW_16[16] = {
    0.0000f, 0.0381f, 0.1464f, 0.3087f, 0.5000f, 0.6913f, 0.8536f, 0.9619f, 
    1.0000f, 0.9619f, 0.8536f, 0.6913f, 0.5000f, 0.3087f, 0.1464f, 0.0381f
};

// 自定义算子：运行时生成 STFT Basis
// 输出形状: [Kernel=16, In=1, Out=18]
static void custom_op_gen_stft_basis(
    struct ggml_tensor * dst,       
    const struct ggml_tensor * a,   
    int ith, int nth, void * userdata) {
    
    const int n_fft = 16;
    const int n_out = n_fft / 2 + 1; // 9 (Nyquist)
    
    // 并行分片
    const int ne = ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth; 
    const int ie0 = dr * ith; 
    const int ie1 = std::min(ie0 + dr, ne);

    float * dst_data = (float *) dst->data;

    for (int i = ie0; i < ie1; i++) {
        // 1. 解析索引
        // Layout: [Kernel(Time), In, Out(Channel)]
        // ne0=16, ne1=1, ne2=18
        // i = out * (1*16) + in * 16 + n
        
        int n = i % n_fft;          // Kernel Time Index (0~15)
        int ch = i / n_fft;         // Output Channel Index (0~17)
        
        // 2. 判断是实部还是虚部
        // 0~8: Real (Cos), 9~17: Imag (Sin)
        int k;      // Frequency Index (0~8)
        bool is_real; 
        
        if (ch < n_out) {
            k = ch;
            is_real = true;
        } else {
            k = ch - n_out;
            is_real = false;
        }

        // 3. 计算值
        float angle = 2.0f * M_PI * k * n / n_fft;
        float w = HANN_WINDOW_16[n];
        
        if (is_real) {
            dst_data[i] = w * std::cos(angle);
        } else {
            // PyTorch stft definition: exp(-j*w*t) -> -sin
            dst_data[i] = w * -std::sin(angle);
        }
    }
}

llm_build_hift::llm_build_hift(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * speech_feat = build_inp_embd(model.f0_classifier_w);
    ggml_set_name(speech_feat, "speech_feat");

    //------------f0_predictor-------------
    ggml_tensor * cur = speech_feat;
    for (int i = 0; i < 10; i += 2) {
        cur = bulid_f0_predictor(cur, model.layers[i].f0_w, model.layers[i].f0_b);
    }
    ggml_tensor * f0_output = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    ggml_tensor * f0 = ggml_mul_mat(ctx0, model.f0_classifier_w, f0_output);
    ggml_tensor * classifier_b = ggml_reshape_3d(ctx0, model.f0_classifier_b, 1, model.f0_classifier_b->ne[0], 1);
    f0 = ggml_add(ctx0, f0, classifier_b);
    ggml_tensor * f0_trans = ggml_reshape_3d(ctx0, f0, f0->ne[1], 1, f0->ne[0]);
    f0_trans = ggml_abs(ctx0, f0_trans);
    ggml_set_name(f0_trans, "f0_predictor_res");
    ggml_tensor * s = ggml_upscale_ext(ctx0, f0_trans, f0_trans->ne[0] * 480, f0_trans->ne[1], f0_trans->ne[2], f0_trans->ne[3], GGML_SCALE_MODE_NEAREST);
    ggml_tensor * s_upsample = ggml_cont(ctx0, ggml_transpose(ctx0, s));
    ggml_set_name(s_upsample, "f0_upsample_res");
    ggml_tensor * s_source = build_m_source(s_upsample, model.m_source_w, model.m_source_b);
    ggml_set_name(s_source, "m_source_res");
    s_source = ggml_cont(ctx0, ggml_transpose(ctx0, s_source));
    ggml_set_name(s_source, "m_source_res_trans");

    //-----------------------decode---------------
    //------------stft-----------
    ggml_tensor * stft_basis = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 16, 1, 18);
    stft_basis = ggml_map_custom1(ctx0, stft_basis, custom_op_gen_stft_basis, GGML_N_TASKS_MAX, NULL);
    stft_basis = ggml_reshape_3d(ctx0, stft_basis, 16, 1, 18);
    const int n_fft = 16;
    const int hop_len = 4;
    ggml_tensor * decode_s_stft = s_source;
    ggml_tensor * stft_out = ggml_conv_1d(ctx0, stft_basis, decode_s_stft, hop_len, 8, 1);
    const int n_freq = n_fft / 2 + 1;
    size_t stride_ch = stft_out->nb[1];
    ggml_tensor * s_stft_real = ggml_view_3d(
        ctx0, stft_out, stft_out->ne[0], n_freq, stft_out->ne[2], stft_out->nb[1], stft_out->nb[2], 0);
    ggml_set_name(s_stft_real, "s_stft_real");
    ggml_tensor * s_stft_imag = ggml_view_3d(
        ctx0, stft_out, stft_out->ne[0], n_freq, stft_out->ne[2], stft_out->nb[1], stft_out->nb[2], n_freq * stft_out->nb[1]);
    ggml_set_name(s_stft_imag, "s_stft_imag");
    ggml_tensor * s_stft = ggml_concat(ctx0, s_stft_real, s_stft_imag, 1);
    ggml_set_name(s_stft, "s_stft");
    //--------stft---------

    //---------conv_pre-----------
    ggml_tensor * conv_res = ggml_conv_1d(ctx0, model.conv_pre_w, speech_feat, 1, 3, 1);
    ggml_tensor * conv_pre_b = ggml_reshape_3d(ctx0, model.conv_pre_b, 1, model.conv_pre_b->ne[0], 1);
    conv_res = ggml_add(ctx0, conv_res, conv_pre_b);
    ggml_set_name(conv_res, "conv_pre_res");
    //---------conv_pre-----------
    
    //--------up_sample---------
    ggml_tensor * up_sample = conv_res;
    int num_upsamples = 3;
    std::vector<int> strides = {8, 5, 3};
    std::vector<int> paddings = {4, 3, 2};
    std::vector<int> source_downs_stride = {15, 3, 1};
    std::vector<int> source_down_padding = {7, 1, 0};
    std::vector<int> source_resblock_kernels = {7, 7, 11};
    for (int i = 0; i < num_upsamples; i++) {
        up_sample = ggml_leaky_relu(ctx0, up_sample, 0.1f, false);
        //-----ups--------
        up_sample = ggml_conv_transpose_1d(ctx0, model.layers[26 + i].ups_w, up_sample, strides[i], 0, 1);
        int p = paddings[i];  // {4, 3, 2}
        int64_t out_len = up_sample->ne[0] - 2 * p;  // 裁掉两端
        int64_t channels = up_sample->ne[1];
        up_sample = ggml_view_2d(ctx0, up_sample, out_len, channels, up_sample->nb[1], p * up_sample->nb[0]);
        up_sample = ggml_cont(ctx0, up_sample);
        up_sample = ggml_add(ctx0, up_sample, ggml_reshape_3d(ctx0, model.layers[26 + i].ups_b, 1, model.layers[26 + i].ups_b->ne[0], 1));
        if (i == num_upsamples - 1) {
            up_sample = ggml_pad_reflect_1d(ctx0, up_sample, 0, 1);
        }
        ggml_set_name(up_sample, ("decode_upsample_" + std::to_string(i)).c_str());

        //-------source_downs------
        ggml_tensor * source_downs = s_stft;
        ggml_tensor * si = ggml_conv_1d(ctx0, model.layers[20 + i].source_downs_w, source_downs, source_downs_stride[i], source_down_padding[i], 1);
        si = ggml_add(ctx0, si, ggml_reshape_3d(ctx0, model.layers[20 + i].source_downs_b, 1, model.layers[20 + i].source_downs_b->ne[0], 1));
        ggml_set_name(si, ("decode_source_downs_" + std::to_string(i)).c_str());
        //------source_downs-------

        //-----source_resblock-----
        ggml_tensor * si_res_blk =  si;
        for(int j = 0; j < 3; j++) {
            int kernel_size = source_resblock_kernels[i];
            int dilation = (j == 0) ? 1 : ((j == 1) ? 3 : 5);
            int padding = (dilation * (kernel_size - 1)) / 2;
            int padding_plain = (1 * (kernel_size - 1)) / 2;
            //-------act1------
            ggml_tensor * act1_res = build_snake(si_res_blk, model.source_resblk_sub_layer[i * 3 + j].source_resblock_act1);
            //-----convs1------
            ggml_tensor * si_res_convs1 = ggml_conv_1d(ctx0, model.source_resblk_sub_layer[i * 3 + j].source_reblock_conv1_w, act1_res, 1, padding, dilation);
            ggml_tensor * b1_reshaped = ggml_reshape_3d(ctx0, model.source_resblk_sub_layer[i * 3 + j].source_reblock_conv1_b, 1, model.source_resblk_sub_layer[i * 3 + j].source_reblock_conv1_b->ne[0], 1);
            si_res_convs1 = ggml_add(ctx0, si_res_convs1, b1_reshaped);
            ggml_tensor * act2_res = build_snake(si_res_convs1, model.source_resblk_sub_layer[i * 3 + j].source_resblock_act2);

            //-----convs2-----
            ggml_tensor * si_res_convs2 = ggml_conv_1d(ctx0, model.source_resblk_sub_layer[i * 3 + j].source_reblock_conv2_w, act2_res, 1, padding_plain, 1);
            ggml_tensor * b2_reshaped = ggml_reshape_3d(ctx0, model.source_resblk_sub_layer[i * 3 + j].source_reblock_conv2_b, 1, model.source_resblk_sub_layer[i * 3 + j].source_reblock_conv2_b->ne[0], 1);
            si_res_convs2 = ggml_add(ctx0, si_res_convs2, b2_reshaped);
            si_res_blk = ggml_add(ctx0, si_res_convs2, si_res_blk);
            ggml_set_name(si_res_blk, ("decode_source_resblocks_" + std::to_string(i) + "_" + std::to_string(j)).c_str());
        }
        up_sample = ggml_add(ctx0, up_sample, si_res_blk);
        ggml_set_name(up_sample, ("decode_after_source_downs_" + std::to_string(i)).c_str());

        //-----resblock-----
        ggml_tensor * xs = NULL;
        std::vector<int> resblk_kernels = {3, 7, 11};
        for(int k = 0; k < 3; k++) {
            int current_kernel_size = resblk_kernels[k];
            int32_t idx = i * 3 + k;
            ggml_tensor * resblk_res = build_res_blk(up_sample, current_kernel_size, idx, model);
            if(xs == NULL) {
                xs = resblk_res;
            } else {
                xs = ggml_add(ctx0, xs, resblk_res);
            }
        }
        ggml_set_name(xs, ("decode_after_resblocks_" + std::to_string(i)).c_str());
        up_sample = ggml_scale(ctx0, xs, 1.0f / 3.0f);
    }

    ggml_tensor * xx = ggml_leaky_relu(ctx0, up_sample, 0.01f, false);
    ggml_set_name(xx, "resblk_leaky_relu");
    xx = ggml_conv_1d(ctx0, model.conv_post_w, xx, 1, 3, 1);
    ggml_tensor * conv_post_b = ggml_reshape_3d(ctx0, model.conv_post_b, 1, model.conv_post_b->ne[0], 1);
    xx = ggml_add(ctx0, xx, conv_post_b);
    ggml_set_name(xx, "conv_post_res");

    res->t_embd = xx;
    // LLAMA_LOG_INFO("res->t_embd ptr: %p\n", res->t_embd);
    ggml_build_forward_expand(gf, xx);
}

