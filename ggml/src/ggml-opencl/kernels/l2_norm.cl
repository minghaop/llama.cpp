kernel void kernel_l2_norm_f32_nd(
    global void * p_src0_base, ulong off_src0_abs,
    global void * p_dst_base,  ulong off_dst_abs,
    int ne00, int ne01, int ne02, int ne03,
    ulong nb00, ulong nb01, ulong nb02, ulong nb03,
    ulong nb10, ulong nb11, ulong nb12, ulong nb13,
    float eps
) {
    int i1 = get_global_id(0);
    int i2 = get_global_id(1);
    int i3 = get_global_id(2);

    if (i1 < ne01 && i2 < ne02 && i3 < ne03) {
        float sum = 0.0f;
        for (int i0 = 0; i0 < ne00; ++i0) {
            ulong src_off = (ulong)i0*nb00 + (ulong)i1*nb01 + (ulong)i2*nb02 + (ulong)i3*nb03;
            global const float * src = (global const float *)((global char *)p_src0_base + off_src0_abs + src_off);
            const float x = *src;
            sum += x * x;
        }

        const float denom = fmax(sqrt(sum), eps);
        const float scale = 1.0f / denom;

        for (int i0 = 0; i0 < ne00; ++i0) {
            ulong src_off = (ulong)i0*nb00 + (ulong)i1*nb01 + (ulong)i2*nb02 + (ulong)i3*nb03;
            ulong dst_off = (ulong)i0*nb10 + (ulong)i1*nb11 + (ulong)i2*nb12 + (ulong)i3*nb13;
            global const float * src = (global const float *)((global char *)p_src0_base + off_src0_abs + src_off);
            global float * dst = (global float *)((global char *)p_dst_base + off_dst_abs + dst_off);
            *dst = (*src) * scale;
        }
    }
}
