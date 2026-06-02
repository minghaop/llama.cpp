kernel void kernel_arange_f32(
    global void * p_dst_base,
    ulong off_dst_abs,
    float start,
    float step,
    int n
) {
    int i = get_global_id(0);
    if (i < n) {
        global float * dst = (global float *)((global char *)p_dst_base + off_dst_abs);
        dst[i] = start + step * (float) i;
    }
}
