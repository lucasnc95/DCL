// adaptive_quadrature_cpu_friendly.cl
// Irregular adaptive numerical integration kernel for heterogeneous load-balancing tests.
// Goal: branch-heavy / loop-irregular workload that tends to be less GPU-friendly than
// regular SIMD kernels, giving CPU devices a non-trivial share in a heterogeneous runtime.
//
// No halo, no swap buffer. One work-item computes one independent adaptive integral.

#pragma OPENCL EXTENSION cl_khr_fp64 : disable

#define Q_STEP 0

inline uint lcg_hash(uint x) {
    x = x * 1664525u + 1013904223u;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return x;
}

inline float f_integrand(float s, float ax, float ay, float t, uint gid) {
    // Oscillatory but bounded integrand. The phase depends on the spatial point and time.
    float w1 = 12.0f + 40.0f * fabs(native_sin(0.21f * ax + 0.017f * t));
    float w2 =  8.0f + 55.0f * fabs(native_cos(0.19f * ay - 0.011f * t));
    float p  = 0.15f + 0.85f * fabs(native_sin(ax * ay + 0.001f * (float)(gid & 1023u)));
    float g  = native_exp(-p * s * s);
    float v  = native_sin(w1 * s + ax) * native_cos(w2 * s * s + ay);
    v += 0.25f * native_sin((w1 + w2) * s * 0.17f + t * 0.003f);
    return g * v;
}

inline float simpson(float a, float b, float ax, float ay, float t, uint gid) {
    float c = 0.5f * (a + b);
    float h = b - a;
    return (h / 6.0f) * (f_integrand(a, ax, ay, t, gid) + 4.0f * f_integrand(c, ax, ay, t, gid) + f_integrand(b, ax, ay, t, gid));
}

__kernel void adaptive_quadrature_cpu_friendly(
    __global float* result,
    __global uint* work_count,
    __global uint* iparams,
    const uint total_points,
    const uint width,
    const uint height,
    const uint max_depth,
    const uint heavy_repeat,
    const float base_tol,
    const float zoom,
    const float center_x,
    const float center_y
) {
    const uint gid = get_global_id(0);
    if (gid >= total_points) return;

    const uint step = iparams[Q_STEP];
    if (get_group_id(0) == 0 && get_local_id(0) == 0) {
        iparams[Q_STEP] = step + 1u;
    }

    const uint x_id = gid % width;
    const uint y_id = gid / width;

    const float x = ((float)x_id / (float)width  - 0.5f) * zoom + center_x;
    const float y = ((float)y_id / (float)height - 0.5f) * zoom + center_y;
    const float t = (float)step;

    // Moving hot region: here the tolerance becomes tighter and recursion depth rises.
    const float hx = 0.38f * native_sin(0.0091f * t);
    const float hy = 0.38f * native_cos(0.0073f * t);
    const float dx = x - hx;
    const float dy = y - hy;
    const float d2 = dx * dx + dy * dy;

    float tol = base_tol;
    if (d2 < 0.010f) tol *= 0.015625f;
    else if (d2 < 0.035f) tol *= 0.0625f;
    else if (d2 < 0.090f) tol *= 0.25f;

    // Per-point deterministic noise: creates warp/wavefront divergence.
    const uint h = lcg_hash(gid ^ (step * 747796405u));
    const uint local_depth = max_depth + (h & 3u);

    // Fixed-size private stack for adaptive Simpson intervals.
    // stack item: [a,b,S,depth]
    float stack_a[96];
    float stack_b[96];
    float stack_s[96];
    uint  stack_d[96];
    int top = 0;

    stack_a[0] = -1.0f;
    stack_b[0] =  1.0f;
    stack_s[0] = simpson(-1.0f, 1.0f, x, y, t, gid);
    stack_d[0] = 0u;

    float integral = 0.0f;
    uint work = 0u;

    while (top >= 0) {
        const float a = stack_a[top];
        const float b = stack_b[top];
        const float s_old = stack_s[top];
        const uint depth = stack_d[top];
        --top;

        const float c = 0.5f * (a + b);
        const float s_left  = simpson(a, c, x, y, t, gid);
        const float s_right = simpson(c, b, x, y, t, gid);
        const float s_new = s_left + s_right;
        const float err = fabs(s_new - s_old);
        work += 1u;

        const int accept = (err < 15.0f * tol) || (depth >= local_depth) || (top > 91);
        if (accept) {
            float val = s_new + (s_new - s_old) / 15.0f;

            // Extra scalar refinement done only for accepted intervals. This keeps the CPU busy
            // but makes GPU lanes diverge because accepted intervals vary strongly by point.
            float q = val;
            uint repeat = heavy_repeat + ((h >> (depth & 15u)) & 15u);
            for (uint k = 0; k < repeat; ++k) {
                float r = (float)(k + 1u) * 0.000137f;
                q += native_sin(q * r + x) * native_cos(y - q * r);
            }
            integral += q;
        } else {
            ++top;
            stack_a[top] = c;
            stack_b[top] = b;
            stack_s[top] = s_right;
            stack_d[top] = depth + 1u;

            ++top;
            stack_a[top] = a;
            stack_b[top] = c;
            stack_s[top] = s_left;
            stack_d[top] = depth + 1u;
        }
    }

    result[gid] = integral;
    work_count[gid] = work;
}
