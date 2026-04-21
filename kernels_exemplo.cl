
__kernel void ocean_jacobi_2d_flat(
    __global const float* in,
    __global float* out,
    const int nx,
    const int ny)
{
    const int gid = get_global_id(0);
    const int total = nx * ny;
    if (gid >= total) return;

    const int x = gid % nx;
    const int y = gid / nx;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
        out[gid] = in[gid];
        return;
    }

    const float center = in[gid];
    const float left   = in[gid - 1];
    const float right  = in[gid + 1];
    const float up     = in[gid - nx];
    const float down   = in[gid + nx];
    out[gid] = 0.2f * (center + left + right + up + down);
}

__kernel void lu_update_2d_heavy_flat(
    __global const float* in,
    __global float* out,
    const int nx,
    const int ny,
    const int flop_repeat)
{
    const int gid = get_global_id(0);
    const int total = nx * ny;
    if (gid >= total) return;

    const int x = gid % nx;
    const int y = gid / nx;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
        out[gid] = in[gid];
        return;
    }

    const float c  = in[gid];
    const float l  = in[gid - 1];
    const float r  = in[gid + 1];
    const float u  = in[gid - nx];
    const float d  = in[gid + nx];
    const float ul = in[gid - nx - 1];
    const float ur = in[gid - nx + 1];
    const float dl = in[gid + nx - 1];
    const float dr = in[gid + nx + 1];

    float v = 0.25f * c
            + 0.125f * (l + r + u + d)
            + 0.0625f * (ul + ur + dl + dr);

    for (int k = 0; k < flop_repeat; ++k) {
        v = v * 1.000001f + 0.000001f;
        v = v * 0.999999f + 0.000001f;
    }
    out[gid] = v;
}

__kernel void map_heavy_1d(
    __global const float* in,
    __global float* out,
    const int n,
    const int flop_repeat)
{
    const int gid = get_global_id(0);
    if (gid >= n) return;

    float v = in[gid] * 1.234567f + 0.765432f;
    for (int k = 0; k < flop_repeat; ++k) {
        v = v * 1.000001f + 0.000001f;
        v = v * 0.999999f + 0.000001f;
    }
    out[gid] = v;
}
