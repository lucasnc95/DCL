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

    const int idx = gid;
    const float center = in[idx];
    const float left   = in[idx - 1];
    const float right  = in[idx + 1];
    const float up     = in[idx - nx];
    const float down   = in[idx + nx];
    printf("Dentro do kernel...");
    out[idx] = 0.2f * (center + left + right + up + down);
}