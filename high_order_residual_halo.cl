// high_order_residual_halo.cl
// 3D high-order finite-difference residual operator.
// Needs halo exchange but does NOT use swap buffers.
//
// Reads a fixed field u and writes residual. The residual is overwritten every iteration.
// Partitions should be z-plane granular; halo width must be radius * nx * ny.

#define S_STEP 0

inline uint idx3(uint x, uint y, uint z, uint nx, uint ny) {
    return z * nx * ny + y * nx + x;
}

__kernel void high_order_residual_halo(
    __global float* residual,
    __global const float* u,
    __global uint* iparams,
    const uint total_points,
    const uint nx,
    const uint ny,
    const uint nz,
    const uint repeat,
    const float alpha,
    const float beta
) {
    const uint gid = get_global_id(0);
    if (gid >= total_points) return;

    const uint step = iparams[S_STEP];
    if (get_group_id(0) == 0 && get_local_id(0) == 0) {
        iparams[S_STEP] = step + 1u;
    }

    const uint xy = nx * ny;
    const uint z = gid / xy;
    const uint rem = gid - z * xy;
    const uint y = rem / nx;
    const uint x = rem - y * nx;

    // 8th-order centered second derivative radius.
    const uint R = 4u;
    if (x < R || x >= nx - R || y < R || y >= ny - R || z < R || z >= nz - R) {
        residual[gid] = 0.0f;
        return;
    }

    const float c0 = -205.0f / 72.0f;
    const float c1 =    8.0f /  5.0f;
    const float c2 =   -1.0f /  5.0f;
    const float c3 =    8.0f / 315.0f;
    const float c4 =   -1.0f / 560.0f;

    const float center = u[gid];
    float acc = 0.0f;
    const float t = (float)step;
    const float omega = 0.25f + 0.05f * native_sin(0.0017f * t);

    for (uint r = 0; r < repeat; ++r) {
        float lapx = c0 * center;
        lapx += c1 * (u[gid - 1u] + u[gid + 1u]);
        lapx += c2 * (u[gid - 2u] + u[gid + 2u]);
        lapx += c3 * (u[gid - 3u] + u[gid + 3u]);
        lapx += c4 * (u[gid - 4u] + u[gid + 4u]);

        float lapy = c0 * center;
        lapy += c1 * (u[gid - nx] + u[gid + nx]);
        lapy += c2 * (u[gid - 2u * nx] + u[gid + 2u * nx]);
        lapy += c3 * (u[gid - 3u * nx] + u[gid + 3u * nx]);
        lapy += c4 * (u[gid - 4u * nx] + u[gid + 4u * nx]);

        float lapz = c0 * center;
        lapz += c1 * (u[gid - xy] + u[gid + xy]);
        lapz += c2 * (u[gid - 2u * xy] + u[gid + 2u * xy]);
        lapz += c3 * (u[gid - 3u * xy] + u[gid + 3u * xy]);
        lapz += c4 * (u[gid - 4u * xy] + u[gid + 4u * xy]);

        float lap = lapx + lapy + lapz;

        // Nonlinear Helmholtz-like residual. The trigonometric work makes the kernel
        // compute-heavy enough to run for long times at large repeat/mesh values.
        float q = alpha * lap - beta * center;
        q += 0.125f * native_sin(center + omega * (float)r);
        q += 0.0625f * native_cos(lap * 0.1f - omega * (float)(r + 1u));
        acc += q * q + 0.000001f * (float)r;
    }

    residual[gid] = acc / (float)repeat;
}
