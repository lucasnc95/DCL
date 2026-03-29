__kernel void jacobi3d_7p(
    __global float* out,
    __global const float* in,
    const int nx,
    const int ny,
    const int nz,
    const int pitch_y,
    const int pitch_z,
    const int flop_repeat
) {
    const int gid = get_global_id(0);

    const int n = nx * ny * nz;
    if (gid >= n) return;

    const int z = gid / (nx * ny);
    const int rem = gid % (nx * ny);
    const int y = rem / nx;
    const int x = rem % nx;

    if (x == 0 || x == nx - 1 ||
        y == 0 || y == ny - 1 ||
        z == 0 || z == nz - 1) {
        out[gid] = in[gid];
        return;
    }

    float center = in[gid];
    float left   = in[gid - 1];
    float right  = in[gid + 1];
    float down   = in[gid - pitch_y];
    float up     = in[gid + pitch_y];
    float back   = in[gid - pitch_z];
    float front  = in[gid + pitch_z];

    float value =
        0.40f * center +
        0.10f * (left + right + down + up + back + front);

    // aumenta intensidade computacional sem alterar o padrão de acesso
    for (int r = 0; r < flop_repeat; ++r) {
        value = 0.99991f * value + 0.00009f * center;
        value = value * 1.00001f - 0.00001f * left;
        value = value + 0.00002f * (right - down + up - back + front);
    }

    out[gid] = value;
}