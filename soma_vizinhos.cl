__kernel void SomaVizinhos1D(
    __global float* out,
    __global const float* in,
    __global const int* params
) {
    const int N = params[0];
    const int gid = (int)get_global_id(0);

    if (gid >= N) {
        return;
    }

    const float left   = (gid > 0)     ? in[gid - 1] : 0.0f;
    const float center = in[gid];
    const float right  = (gid < N - 1) ? in[gid + 1] : 0.0f;

    out[gid] = left + center + right;
}