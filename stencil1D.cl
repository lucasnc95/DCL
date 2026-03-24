__kernel void Stencil1D(
    __global const float* in,
    __global float* out,
    __global const int* params
) {
    // params[0] = N total
    const int N = params[0];

    const size_t gid = get_global_id(0);

    if ((int)gid >= N) {
        return;
    }

    // bordas globais
    if (gid == 0 || gid == (size_t)(N - 1)) {
        out[gid] = in[gid];
        return;
    }

    const float left   = in[gid - 1];
    const float center = in[gid];
    const float right  = in[gid + 1];

    out[gid] = (left + center + right) / 3.0f;
}