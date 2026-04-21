#include <CL/cl.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <climits>

#ifdef USE_DCL
#include "dcl/runtime.hpp"
#endif

namespace {

enum class ImplKind { dcl_impl, mpiocl };
enum class DeviceFilter { cpu, gpu, all };
enum class KernelKind { ocean, lu, map };

struct Args {
    ImplKind impl = ImplKind::mpiocl;
    KernelKind kernel = KernelKind::ocean;
    DeviceFilter device = DeviceFilter::all;
    int iterations = 100;
    int nx = 1024;
    int ny = 1024;
    int n = 1 << 24;
    int flop_repeat = 64;
    std::string kernel_file = "kernels.cl";
};

[[noreturn]] inline void fail(const std::string& msg) {
    throw std::runtime_error(msg);
}

inline void check_cl(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        std::ostringstream oss;
        oss << what << " failed with OpenCL error " << err;
        fail(oss.str());
    }
}

inline void check_mpi(int err, const char* what) {
    if (err != MPI_SUCCESS) {
        char buf[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(err, buf, &len);
        std::ostringstream oss;
        oss << what << " failed with MPI error: " << std::string(buf, len);
        fail(oss.str());
    }
}

inline std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline ImplKind parse_impl(const std::string& s) {
    const std::string v = to_lower(s);
    if (v == "dcl") return ImplKind::dcl_impl;
    if (v == "mpiocl") return ImplKind::mpiocl;
    fail("Unknown --impl value: " + s);
}

inline DeviceFilter parse_device(const std::string& s) {
    const std::string v = to_lower(s);
    if (v == "cpu") return DeviceFilter::cpu;
    if (v == "gpu") return DeviceFilter::gpu;
    if (v == "all") return DeviceFilter::all;
    fail("Unknown --device value: " + s);
}

inline KernelKind parse_kernel(const std::string& s) {
    const std::string v = to_lower(s);
    if (v == "ocean") return KernelKind::ocean;
    if (v == "lu") return KernelKind::lu;
    if (v == "map") return KernelKind::map;
    fail("Unknown --kernel value: " + s);
}

inline std::string kernel_name(KernelKind k) {
    switch (k) {
        case KernelKind::ocean: return "ocean_jacobi_2d_flat";
        case KernelKind::lu:    return "lu_update_2d_heavy_flat";
        case KernelKind::map:   return "map_heavy_1d";
    }
    return "";
}

inline Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) fail(std::string("Missing value for ") + name);
            return argv[++i];
        };
        if (key == "--impl") a.impl = parse_impl(need("--impl"));
        else if (key == "--kernel") a.kernel = parse_kernel(need("--kernel"));
        else if (key == "--device") a.device = parse_device(need("--device"));
        else if (key == "--iters") a.iterations = std::stoi(need("--iters"));
        else if (key == "--nx") a.nx = std::stoi(need("--nx"));
        else if (key == "--ny") a.ny = std::stoi(need("--ny"));
        else if (key == "--n") a.n = std::stoi(need("--n"));
        else if (key == "--flop-repeat") a.flop_repeat = std::stoi(need("--flop-repeat"));
        else if (key == "--kernel-file") a.kernel_file = need("--kernel-file");
        else fail("Unknown argument: " + key);
    }
    return a;
}

inline const char* kernels_source() {
    return R"CLC(
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
)CLC";
}

inline void write_kernel_file(const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs) fail("Cannot create kernel file: " + path);
    ofs << kernels_source();
}

struct OclDeviceCtx {
    cl_device_id dev = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_program prog = nullptr;
    std::string name;
    cl_uint compute_units = 0;
};

inline std::vector<OclDeviceCtx> discover_opencl_devices(DeviceFilter filter, const std::string& src) {
    cl_uint nplat = 0;
    check_cl(clGetPlatformIDs(0, nullptr, &nplat), "clGetPlatformIDs(count)");
    std::vector<cl_platform_id> plats(nplat);
    check_cl(clGetPlatformIDs(nplat, plats.data(), nullptr), "clGetPlatformIDs(list)");

    const cl_device_type dtype = (filter == DeviceFilter::cpu) ? CL_DEVICE_TYPE_CPU
                               : (filter == DeviceFilter::gpu) ? CL_DEVICE_TYPE_GPU
                               : CL_DEVICE_TYPE_ALL;

    std::vector<OclDeviceCtx> out;
    for (cl_platform_id p : plats) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(p, dtype, 0, nullptr, &ndev) != CL_SUCCESS || ndev == 0) continue;
        std::vector<cl_device_id> devs(ndev);
        check_cl(clGetDeviceIDs(p, dtype, ndev, devs.data(), nullptr), "clGetDeviceIDs");
        for (cl_device_id d : devs) {
            size_t name_sz = 0;
            check_cl(clGetDeviceInfo(d, CL_DEVICE_NAME, 0, nullptr, &name_sz), "clGetDeviceInfo(name size)");
            std::string name(name_sz, '\0');
            check_cl(clGetDeviceInfo(d, CL_DEVICE_NAME, name_sz, name.data(), nullptr), "clGetDeviceInfo(name)");
            if (!name.empty() && name.back() == '\0') name.pop_back();
            cl_uint cu = 0;
            check_cl(clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr), "clGetDeviceInfo(cu)");

            cl_int err = CL_SUCCESS;
            cl_context ctx = clCreateContext(nullptr, 1, &d, nullptr, nullptr, &err);
            check_cl(err, "clCreateContext");
#if CL_TARGET_OPENCL_VERSION >= 200
            cl_command_queue q = clCreateCommandQueueWithProperties(ctx, d, nullptr, &err);
#else
            cl_command_queue q = clCreateCommandQueue(ctx, d, 0, &err);
#endif
            check_cl(err, "clCreateCommandQueue");
            const char* csrc = src.c_str();
            const size_t len = src.size();
            cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &len, &err);
            check_cl(err, "clCreateProgramWithSource");
            err = clBuildProgram(prog, 1, &d, "", nullptr, nullptr);
            if (err != CL_SUCCESS) {
                size_t log_sz = 0;
                clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
                std::string log(log_sz, '\0');
                clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
                std::cerr << log << std::endl;
                check_cl(err, "clBuildProgram");
            }
            out.push_back({d, ctx, q, prog, name, cu});
        }
    }
    if (out.empty()) fail("No OpenCL devices found for selected filter");
    return out;
}

inline void release_opencl_devices(std::vector<OclDeviceCtx>& devs) {
    for (OclDeviceCtx& d : devs) {
        if (d.prog) clReleaseProgram(d.prog);
        if (d.q) clReleaseCommandQueue(d.q);
        if (d.ctx) clReleaseContext(d.ctx);
        d = {};
    }
}

inline void print_devices_all_ranks_opencl(const std::vector<OclDeviceCtx>& devs, int rank, int world) {
    for (int r = 0; r < world; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r) {
            std::cout << "=== rank " << rank << " local devices ===\n";
            for (std::size_t i = 0; i < devs.size(); ++i) {
                std::cout << "local=" << i
                          << " name=\"" << devs[i].name << "\""
                          << " compute_units=" << devs[i].compute_units << "\n";
            }
            std::cout << std::flush;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

inline std::vector<std::pair<std::size_t, std::size_t>> split_even(std::size_t total, std::size_t parts) {
    std::vector<std::pair<std::size_t, std::size_t>> out(parts);
    const std::size_t base = total / parts;
    const std::size_t rem = total % parts;
    std::size_t off = 0;
    for (std::size_t i = 0; i < parts; ++i) {
        const std::size_t cnt = base + (i < rem ? 1 : 0);
        out[i] = {off, cnt};
        off += cnt;
    }
    return out;
}

inline std::vector<std::size_t> sample_indices(std::size_t total) {
    std::vector<std::size_t> idx;
    if (total == 0) return idx;
    idx.push_back(0);
    idx.push_back(std::min<std::size_t>(1, total - 1));
    idx.push_back(std::min<std::size_t>(2, total - 1));
    idx.push_back(total / 7);
    idx.push_back(total / 5);
    idx.push_back(total / 3);
    idx.push_back(total / 2);
    idx.push_back((2 * total) / 3);
    idx.push_back((4 * total) / 5);
    idx.push_back(total - 1);
    std::sort(idx.begin(), idx.end());
    idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
    return idx;
}

inline void print_samples_rank0(const std::vector<float>& data,
                                const std::vector<std::size_t>& idx,
                                const std::string& label,
                                int rank) {
    if (rank != 0) return;
    std::cout << label << "\n";
    for (std::size_t i : idx) {
        std::cout << "  [" << i << "] = " << std::fixed << std::setprecision(6) << data[i] << "\n";
    }
}

inline void print_exec_characteristics_rank0(const Args& args, int rank) {
    if (rank != 0) return;
    std::cout << "=== execution ===\n";
    std::cout << "impl=" << (args.impl == ImplKind::dcl_impl ? "dcl" : "mpiocl") << "\n";
    std::cout << "kernel=" << kernel_name(args.kernel) << "\n";
    std::cout << "device=" << (args.device == DeviceFilter::cpu ? "cpu" : args.device == DeviceFilter::gpu ? "gpu" : "all") << "\n";
    std::cout << "iterations=" << args.iterations << "\n";
    std::cout << "OpenCL local work-group = auto (nullptr / nullopt)\n";
    if (args.kernel == KernelKind::ocean || args.kernel == KernelKind::lu) {
        std::cout << "nx=" << args.nx << " ny=" << args.ny << "\n";
        if (args.kernel == KernelKind::lu) std::cout << "flop_repeat=" << args.flop_repeat << "\n";
    } else {
        std::cout << "n=" << args.n << " flop_repeat=" << args.flop_repeat << "\n";
    }
}

struct Partition1D {
    int global_device_index = -1;
    int owning_rank = -1;
    int local_index = -1;
    std::size_t global_offset = 0;
    std::size_t element_count = 0;
};

struct PlainKernelEvent {
    std::size_t local_device;
    cl_event event;
};

inline std::vector<int> build_all_device_counts(int local_count, MPI_Comm comm) {
    int size = 1;
    MPI_Comm_size(comm, &size);
    std::vector<int> counts(size, 0);
    check_mpi(MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm), "MPI_Allgather(device_counts)");
    return counts;
}

inline int owner_rank_of_global_device(const std::vector<int>& counts, int g) {
    int acc = 0;
    for (int r = 0; r < static_cast<int>(counts.size()); ++r) {
        if (g < acc + counts[r]) return r;
        acc += counts[r];
    }
    return static_cast<int>(counts.size()) - 1;
}

inline int local_index_of_global_device(const std::vector<int>& counts, int self_rank, int g) {
    int acc = 0;
    for (int r = 0; r < static_cast<int>(counts.size()); ++r) {
        if (g < acc + counts[r]) return (r == self_rank) ? (g - acc) : -1;
        acc += counts[r];
    }
    return -1;
}

inline std::vector<Partition1D> build_equal_partitions(std::size_t global_elements,
                                                       const std::vector<int>& all_device_counts,
                                                       int self_rank) {
    int total_devices = 0;
    for (int c : all_device_counts) total_devices += c;
    if (total_devices <= 0) fail("No global devices available");

    std::vector<Partition1D> out;
    out.reserve(static_cast<std::size_t>(total_devices));

    const std::size_t base = global_elements / static_cast<std::size_t>(total_devices);
    const std::size_t rem = global_elements % static_cast<std::size_t>(total_devices);

    std::size_t cur = 0;
    for (int g = 0; g < total_devices; ++g) {
        const std::size_t len = base + (static_cast<std::size_t>(g) < rem ? 1u : 0u);
        Partition1D p;
        p.global_device_index = g;
        p.owning_rank = owner_rank_of_global_device(all_device_counts, g);
        p.local_index = local_index_of_global_device(all_device_counts, self_rank, g);
        p.global_offset = cur;
        p.element_count = len;
        out.push_back(p);
        cur += len;
    }
    return out;
}

inline bool needs_halo_exchange_plain(const std::vector<Partition1D>& parts,
                                      std::size_t halo,
                                      bool has_halo_fields) {
    if (halo == 0) return false;
    if (!has_halo_fields) return false;
    return parts.size() > 1;
}

inline void finalize_kernel_events_plain(std::vector<PlainKernelEvent>& kernel_events) {
    for (std::size_t i = 0; i < kernel_events.size(); ++i) {
        if (kernel_events[i].event != nullptr) {
            check_cl(clWaitForEvents(1, &kernel_events[i].event), "clWaitForEvents(finalize_kernel_events_plain)");
            clReleaseEvent(kernel_events[i].event);
            kernel_events[i].event = nullptr;
        }
    }
    kernel_events.clear();
}

inline void enqueue_full_phase_plain(const std::vector<Partition1D>& parts,
                                     int rank,
                                     const std::vector<OclDeviceCtx>& devs,
                                     cl_kernel* kernels,
                                     cl_mem* inb,
                                     cl_mem* outb,
                                     int nx,
                                     int ny,
                                     bool heavy,
                                     int flop_repeat,
                                     std::vector<PlainKernelEvent>& kernel_events) {
    kernel_events.clear();
    for (std::size_t p = 0; p < parts.size(); ++p) {
        const Partition1D& dp = parts[p];
        if (dp.owning_rank != rank || dp.local_index < 0 || dp.element_count == 0) continue;
        const std::size_t d = static_cast<std::size_t>(dp.local_index);
        check_cl(clSetKernelArg(kernels[d], 0, sizeof(cl_mem), &inb[d]), "clSetKernelArg(full arg0)");
        check_cl(clSetKernelArg(kernels[d], 1, sizeof(cl_mem), &outb[d]), "clSetKernelArg(full arg1)");
        check_cl(clSetKernelArg(kernels[d], 2, sizeof(int), &nx), "clSetKernelArg(full arg2)");
        check_cl(clSetKernelArg(kernels[d], 3, sizeof(int), &ny), "clSetKernelArg(full arg3)");
        if (heavy) check_cl(clSetKernelArg(kernels[d], 4, sizeof(int), &flop_repeat), "clSetKernelArg(full arg4)");
        const std::size_t gwo = dp.global_offset;
        const std::size_t gws = dp.element_count;
        cl_event ev = nullptr;
        check_cl(clEnqueueNDRangeKernel(devs[d].q, kernels[d], 1, &gwo, &gws, nullptr, 0, nullptr, &ev),
                 "clEnqueueNDRangeKernel(full)");
        kernel_events.push_back({d, ev});
    }
}

inline void enqueue_interior_phase_plain(const std::vector<Partition1D>& parts,
                                         int rank,
                                         const std::vector<OclDeviceCtx>& devs,
                                         cl_kernel* kernels,
                                         cl_mem* inb,
                                         cl_mem* outb,
                                         int nx,
                                         int ny,
                                         bool heavy,
                                         int flop_repeat,
                                         std::size_t halo,
                                         std::vector<PlainKernelEvent>& kernel_events) {
    kernel_events.clear();
    for (std::size_t p = 0; p < parts.size(); ++p) {
        const Partition1D& dp = parts[p];
        if (dp.owning_rank != rank || dp.local_index < 0) continue;
        if (dp.element_count <= 2 * halo) continue;
        const std::size_t d = static_cast<std::size_t>(dp.local_index);
        check_cl(clSetKernelArg(kernels[d], 0, sizeof(cl_mem), &inb[d]), "clSetKernelArg(interior arg0)");
        check_cl(clSetKernelArg(kernels[d], 1, sizeof(cl_mem), &outb[d]), "clSetKernelArg(interior arg1)");
        check_cl(clSetKernelArg(kernels[d], 2, sizeof(int), &nx), "clSetKernelArg(interior arg2)");
        check_cl(clSetKernelArg(kernels[d], 3, sizeof(int), &ny), "clSetKernelArg(interior arg3)");
        if (heavy) check_cl(clSetKernelArg(kernels[d], 4, sizeof(int), &flop_repeat), "clSetKernelArg(interior arg4)");
        const std::size_t gwo = dp.global_offset + halo;
        const std::size_t gws = dp.element_count - 2 * halo;
        cl_event ev = nullptr;
        check_cl(clEnqueueNDRangeKernel(devs[d].q, kernels[d], 1, &gwo, &gws, nullptr, 0, nullptr, &ev),
                 "clEnqueueNDRangeKernel(interior)");
        kernel_events.push_back({d, ev});
    }
}

inline void enqueue_border_phase_plain(const std::vector<Partition1D>& parts,
                                       int rank,
                                       const std::vector<OclDeviceCtx>& devs,
                                       cl_kernel* kernels,
                                       cl_mem* inb,
                                       cl_mem* outb,
                                       int nx,
                                       int ny,
                                       bool heavy,
                                       int flop_repeat,
                                       std::size_t halo,
                                       std::vector<PlainKernelEvent>& kernel_events) {
    kernel_events.clear();
    for (std::size_t p = 0; p < parts.size(); ++p) {
        const Partition1D& dp = parts[p];
        if (dp.owning_rank != rank || dp.local_index < 0 || dp.element_count == 0) continue;
        const std::size_t d = static_cast<std::size_t>(dp.local_index);
        check_cl(clSetKernelArg(kernels[d], 0, sizeof(cl_mem), &inb[d]), "clSetKernelArg(border arg0)");
        check_cl(clSetKernelArg(kernels[d], 1, sizeof(cl_mem), &outb[d]), "clSetKernelArg(border arg1)");
        check_cl(clSetKernelArg(kernels[d], 2, sizeof(int), &nx), "clSetKernelArg(border arg2)");
        check_cl(clSetKernelArg(kernels[d], 3, sizeof(int), &ny), "clSetKernelArg(border arg3)");
        if (heavy) check_cl(clSetKernelArg(kernels[d], 4, sizeof(int), &flop_repeat), "clSetKernelArg(border arg4)");

        const std::size_t left_count = std::min(halo, dp.element_count);
        const std::size_t right_count = (dp.element_count > halo) ? std::min(halo, dp.element_count - left_count) : 0;

        if (left_count > 0) {
            const std::size_t gwo = dp.global_offset;
            const std::size_t gws = left_count;
            cl_event ev = nullptr;
            check_cl(clEnqueueNDRangeKernel(devs[d].q, kernels[d], 1, &gwo, &gws, nullptr, 0, nullptr, &ev),
                     "clEnqueueNDRangeKernel(border-left)");
            kernel_events.push_back({d, ev});
        }
        if (right_count > 0 && dp.element_count > left_count) {
            const std::size_t gwo = dp.global_offset + dp.element_count - right_count;
            const std::size_t gws = right_count;
            cl_event ev = nullptr;
            check_cl(clEnqueueNDRangeKernel(devs[d].q, kernels[d], 1, &gwo, &gws, nullptr, 0, nullptr, &ev),
                     "clEnqueueNDRangeKernel(border-right)");
            kernel_events.push_back({d, ev});
        }
    }
}

inline void exchange_halos_plain(const std::vector<Partition1D>& parts,
                                 int rank,
                                 MPI_Comm comm,
                                 const std::vector<OclDeviceCtx>& devs,
                                 cl_mem* field_bufs,
                                 std::size_t halo,
                                 std::size_t elem_bytes) {
    if (halo == 0 || parts.size() < 2) return;
    const std::size_t halo_bytes = halo * elem_bytes;

    std::vector<unsigned char> send_lr(halo_bytes), send_rl(halo_bytes), recv_lr(halo_bytes), recv_rl(halo_bytes);

    for (std::size_t b = 0; b + 1 < parts.size(); ++b) {
        const Partition1D& left = parts[b];
        const Partition1D& right = parts[b + 1];
        if (left.element_count < halo || right.element_count < halo) continue;

        const std::size_t left_border_off = (left.global_offset + left.element_count - halo) * elem_bytes;
        const std::size_t left_halo_off   = (left.global_offset + left.element_count) * elem_bytes;
        const std::size_t right_border_off = right.global_offset * elem_bytes;
        const std::size_t right_halo_off   = (right.global_offset - halo) * elem_bytes;

        if (left.owning_rank == rank) {
            const std::size_t dl = static_cast<std::size_t>(left.local_index);
            cl_event read_ev = nullptr;
            check_cl(clEnqueueReadBuffer(devs[dl].q, field_bufs[dl], CL_FALSE, left_border_off, halo_bytes, send_lr.data(), 0, nullptr, &read_ev),
                     "clEnqueueReadBuffer(halo lr)");
            check_cl(clWaitForEvents(1, &read_ev), "clWaitForEvents(halo lr)");
            clReleaseEvent(read_ev);
        }
        if (right.owning_rank == rank) {
            const std::size_t dr = static_cast<std::size_t>(right.local_index);
            cl_event read_ev = nullptr;
            check_cl(clEnqueueReadBuffer(devs[dr].q, field_bufs[dr], CL_FALSE, right_border_off, halo_bytes, send_rl.data(), 0, nullptr, &read_ev),
                     "clEnqueueReadBuffer(halo rl)");
            check_cl(clWaitForEvents(1, &read_ev), "clWaitForEvents(halo rl)");
            clReleaseEvent(read_ev);
        }

        if (left.owning_rank == rank && right.owning_rank == rank) {
            const std::size_t dl = static_cast<std::size_t>(left.local_index);
            const std::size_t dr = static_cast<std::size_t>(right.local_index);
            check_cl(clEnqueueWriteBuffer(devs[dr].q, field_bufs[dr], CL_FALSE, right_halo_off, halo_bytes, send_lr.data(), 0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(local lr)");
            check_cl(clEnqueueWriteBuffer(devs[dl].q, field_bufs[dl], CL_FALSE, left_halo_off, halo_bytes, send_rl.data(), 0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(local rl)");
            continue;
        }

        const int tag_lr = 10000 + static_cast<int>(b);
        const int tag_rl = 20000 + static_cast<int>(b);

        if (left.owning_rank == rank) {
            MPI_Request reqs[2];
            check_mpi(MPI_Isend(send_lr.data(), static_cast<int>(halo_bytes), MPI_BYTE, right.owning_rank, tag_lr, comm, &reqs[0]), "MPI_Isend(lr)");
            check_mpi(MPI_Irecv(recv_rl.data(), static_cast<int>(halo_bytes), MPI_BYTE, right.owning_rank, tag_rl, comm, &reqs[1]), "MPI_Irecv(rl)");
            check_mpi(MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE), "MPI_Waitall(left)");
            const std::size_t dl = static_cast<std::size_t>(left.local_index);
            check_cl(clEnqueueWriteBuffer(devs[dl].q, field_bufs[dl], CL_FALSE, left_halo_off, halo_bytes, recv_rl.data(), 0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(remote rl)");
        }
        if (right.owning_rank == rank) {
            MPI_Request reqs[2];
            check_mpi(MPI_Irecv(recv_lr.data(), static_cast<int>(halo_bytes), MPI_BYTE, left.owning_rank, tag_lr, comm, &reqs[0]), "MPI_Irecv(lr)");
            check_mpi(MPI_Isend(send_rl.data(), static_cast<int>(halo_bytes), MPI_BYTE, left.owning_rank, tag_rl, comm, &reqs[1]), "MPI_Isend(rl)");
            check_mpi(MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE), "MPI_Waitall(right)");
            const std::size_t dr = static_cast<std::size_t>(right.local_index);
            check_cl(clEnqueueWriteBuffer(devs[dr].q, field_bufs[dr], CL_FALSE, right_halo_off, halo_bytes, recv_lr.data(), 0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(remote lr)");
        }
    }
}

inline void gather_field_plain(const std::vector<Partition1D>& parts,
                               int rank,
                               MPI_Comm comm,
                               const std::vector<OclDeviceCtx>& devs,
                               cl_mem* field_bufs,
                               std::size_t global_elements,
                               std::size_t elem_bytes,
                               std::vector<unsigned char>& root_out) {
    const std::size_t total_bytes = global_elements * elem_bytes;
    std::vector<unsigned char> local_image(total_bytes, 0);
    std::vector<cl_event> read_events;

    for (std::size_t p = 0; p < parts.size(); ++p) {
        const Partition1D& dp = parts[p];
        if (dp.owning_rank != rank || dp.local_index < 0) continue;
        const std::size_t d = static_cast<std::size_t>(dp.local_index);
        const std::size_t off = dp.global_offset * elem_bytes;
        const std::size_t len = dp.element_count * elem_bytes;
        cl_event ev = nullptr;
        check_cl(clEnqueueReadBuffer(devs[d].q, field_bufs[d], CL_FALSE, off, len, local_image.data() + off, 0, nullptr, &ev),
                 "clEnqueueReadBuffer(gather plain)");
        read_events.push_back(ev);
    }

    for (cl_event ev : read_events) {
        check_cl(clWaitForEvents(1, &ev), "clWaitForEvents(gather plain)");
        clReleaseEvent(ev);
    }

    if (rank == 0) root_out.assign(total_bytes, 0);

    std::size_t offset = 0;
    while (offset < total_bytes) {
        const std::size_t rem = total_bytes - offset;
        const int chunk = static_cast<int>(std::min<std::size_t>(rem, static_cast<std::size_t>(INT_MAX)));
        check_mpi(MPI_Reduce(local_image.data() + offset,
                             rank == 0 ? (root_out.data() + offset) : nullptr,
                             chunk,
                             MPI_BYTE,
                             MPI_BOR,
                             0,
                             comm),
                  "MPI_Reduce(gather plain)");
        offset += static_cast<std::size_t>(chunk);
    }
}

inline void run_stencil_mpiocl_dcl_like(const Args& args, bool heavy, int rank, int world) {
    const int nx = args.nx;
    const int ny = args.ny;
    const std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const std::size_t halo = static_cast<std::size_t>(nx);

    const std::string src = kernels_source();
    auto devs = discover_opencl_devices(args.device, src);
    print_devices_all_ranks_opencl(devs, rank, world);
    print_exec_characteristics_rank0(args, rank);

    const std::vector<int> all_device_counts = build_all_device_counts(static_cast<int>(devs.size()), MPI_COMM_WORLD);
    const std::vector<Partition1D> parts = build_equal_partitions(total, all_device_counts, rank);
    const std::size_t global_parts = parts.size();
    const bool use_halo = needs_halo_exchange_plain(parts, halo, true);

    std::vector<float> a(total, 0.0f), b(total, 0.0f);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            a[static_cast<std::size_t>(y) * nx + x] = 0.001f * static_cast<float>((x + 3 * y) % 1000);
        }
    }
    b = a;

    std::vector<cl_kernel> kernels(devs.size(), nullptr);
    std::vector<cl_mem> a_dev(devs.size(), nullptr);
    std::vector<cl_mem> b_dev(devs.size(), nullptr);

    for (std::size_t d = 0; d < devs.size(); ++d) {
        cl_int err = CL_SUCCESS;
        kernels[d] = clCreateKernel(devs[d].prog, heavy ? "lu_update_2d_heavy_flat" : "ocean_jacobi_2d_flat", &err);
        check_cl(err, "clCreateKernel(stencil plain)");
        a_dev[d] = clCreateBuffer(devs[d].ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total * sizeof(float), a.data(), &err);
        check_cl(err, "clCreateBuffer(a_dev)");
        b_dev[d] = clCreateBuffer(devs[d].ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total * sizeof(float), b.data(), &err);
        check_cl(err, "clCreateBuffer(b_dev)");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    bool a_to_b = true;
    std::vector<PlainKernelEvent> kernel_events;

    for (int it = 0; it < args.iterations; ++it) {
        cl_mem* inb = a_to_b ? a_dev.data() : b_dev.data();
        cl_mem* outb = a_to_b ? b_dev.data() : a_dev.data();

        if (global_parts <= 1) {
            enqueue_full_phase_plain(parts, rank, devs, kernels.data(), inb, outb, nx, ny, heavy, args.flop_repeat, kernel_events);
            finalize_kernel_events_plain(kernel_events);
        } else if (!use_halo) {
            enqueue_full_phase_plain(parts, rank, devs, kernels.data(), inb, outb, nx, ny, heavy, args.flop_repeat, kernel_events);
            finalize_kernel_events_plain(kernel_events);
        } else {
            enqueue_interior_phase_plain(parts, rank, devs, kernels.data(), inb, outb, nx, ny, heavy, args.flop_repeat, halo, kernel_events);
            exchange_halos_plain(parts, rank, MPI_COMM_WORLD, devs, inb, halo, sizeof(float));
            finalize_kernel_events_plain(kernel_events);
            enqueue_border_phase_plain(parts, rank, devs, kernels.data(), inb, outb, nx, ny, heavy, args.flop_repeat, halo, kernel_events);
            finalize_kernel_events_plain(kernel_events);
        }

        a_to_b = !a_to_b;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();

    std::vector<unsigned char> gathered_bytes;
    gather_field_plain(parts, rank, MPI_COMM_WORLD, devs, a_to_b ? a_dev.data() : b_dev.data(), total, sizeof(float), gathered_bytes);

    if (rank == 0) {
        std::vector<float> out(total, 0.0f);
        std::memcpy(out.data(), gathered_bytes.data(), total * sizeof(float));
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << (heavy ? "LU MPI+OCL(DCL-like)\n" : "OCEAN MPI+OCL(DCL-like)\n");
        std::cout << "tempo_total_s=" << sec << "\n";
        std::cout << "tempo_medio_iter_s=" << (sec / args.iterations) << "\n";
        print_samples_rank0(out, sample_indices(out.size()), "samples", rank);
    }

    for (std::size_t d = 0; d < devs.size(); ++d) {
        if (a_dev[d]) clReleaseMemObject(a_dev[d]);
        if (b_dev[d]) clReleaseMemObject(b_dev[d]);
        if (kernels[d]) clReleaseKernel(kernels[d]);
    }
    release_opencl_devices(devs);
}

inline void run_map_mpiocl_dcl_like(const Args& args, int rank, int world) {
    const std::size_t total = static_cast<std::size_t>(args.n);

    const std::string src = kernels_source();
    auto devs = discover_opencl_devices(args.device, src);
    print_devices_all_ranks_opencl(devs, rank, world);
    print_exec_characteristics_rank0(args, rank);

    const std::vector<int> all_device_counts = build_all_device_counts(static_cast<int>(devs.size()), MPI_COMM_WORLD);
    const std::vector<Partition1D> parts = build_equal_partitions(total, all_device_counts, rank);

    std::vector<float> in(total, 0.0f), out(total, 0.0f);
    for (std::size_t i = 0; i < total; ++i) {
        in[i] = 0.001f * static_cast<float>(i);
    }

    std::vector<cl_kernel> kernels(devs.size(), nullptr);
    std::vector<cl_mem> in_dev(devs.size(), nullptr);
    std::vector<cl_mem> out_dev(devs.size(), nullptr);

    for (std::size_t d = 0; d < devs.size(); ++d) {
        cl_int err = CL_SUCCESS;
        kernels[d] = clCreateKernel(devs[d].prog, "map_heavy_1d", &err);
        check_cl(err, "clCreateKernel(map plain)");
        in_dev[d] = clCreateBuffer(devs[d].ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total * sizeof(float), in.data(), &err);
        check_cl(err, "clCreateBuffer(in_dev)");
        out_dev[d] = clCreateBuffer(devs[d].ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total * sizeof(float), out.data(), &err);
        check_cl(err, "clCreateBuffer(out_dev)");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    std::vector<PlainKernelEvent> kernel_events;
    for (int it = 0; it < args.iterations; ++it) {
        kernel_events.clear();
        for (std::size_t p = 0; p < parts.size(); ++p) {
            const Partition1D& dp = parts[p];
            if (dp.owning_rank != rank || dp.local_index < 0 || dp.element_count == 0) continue;
            const std::size_t d = static_cast<std::size_t>(dp.local_index);
            check_cl(clSetKernelArg(kernels[d], 0, sizeof(cl_mem), &in_dev[d]), "map arg0");
            check_cl(clSetKernelArg(kernels[d], 1, sizeof(cl_mem), &out_dev[d]), "map arg1");
            const int part_n = static_cast<int>(dp.element_count);
            check_cl(clSetKernelArg(kernels[d], 2, sizeof(int), &part_n), "map arg2");
            check_cl(clSetKernelArg(kernels[d], 3, sizeof(int), &args.flop_repeat), "map arg3");
            const std::size_t gwo = dp.global_offset;
            const std::size_t gws = dp.element_count;
            cl_event ev = nullptr;
            check_cl(clEnqueueNDRangeKernel(devs[d].q, kernels[d], 1, &gwo, &gws, nullptr, 0, nullptr, &ev),
                     "clEnqueueNDRangeKernel(map plain)");
            kernel_events.push_back({d, ev});
        }
        finalize_kernel_events_plain(kernel_events);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();

    std::vector<unsigned char> gathered_bytes;
    gather_field_plain(parts, rank, MPI_COMM_WORLD, devs, out_dev.data(), total, sizeof(float), gathered_bytes);

    if (rank == 0) {
        std::vector<float> gathered(total, 0.0f);
        std::memcpy(gathered.data(), gathered_bytes.data(), total * sizeof(float));
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "MAP MPI+OCL(DCL-like)\n";
        std::cout << "tempo_total_s=" << sec << "\n";
        std::cout << "tempo_medio_iter_s=" << (sec / args.iterations) << "\n";
        print_samples_rank0(gathered, sample_indices(gathered.size()), "samples", rank);
    }

    for (std::size_t d = 0; d < devs.size(); ++d) {
        if (in_dev[d]) clReleaseMemObject(in_dev[d]);
        if (out_dev[d]) clReleaseMemObject(out_dev[d]);
        if (kernels[d]) clReleaseKernel(kernels[d]);
    }
    release_opencl_devices(devs);
}

inline void run_ocean_mpiocl_dcl_like(const Args& args, int rank, int world) {
    run_stencil_mpiocl_dcl_like(args, false, rank, world);
}

inline void run_lu_mpiocl_dcl_like(const Args& args, int rank, int world) {
    run_stencil_mpiocl_dcl_like(args, true, rank, world);
}

#ifdef USE_DCL

inline void print_devices_all_ranks_dcl(dcl::Runtime& runtime, int rank, int world) {
    const std::vector<dcl::DeviceInfo>& devs = runtime.devices();
    for (int r = 0; r < world; ++r) {
        //runtime.synchronize(true);
        if (rank == r) {
            std::cout << "=== rank " << rank << " local devices ===\n";
            for (std::size_t i = 0; i < devs.size(); ++i) {
                std::cout << "global=" << devs[i].global_index
                          << " local=" << devs[i].local_index
                          << " name=\"" << devs[i].name << "\""
                          << " compute_units=" << devs[i].compute_units << "\n";
            }
            std::cout << std::flush;
        }
    }
  //  runtime.synchronize(true);
}

inline void run_stencil_dcl(const Args& args, bool heavy, int argc, char** argv, int rank, int world) {
    using clock_t = std::chrono::steady_clock;
    const int nx = args.nx;
    const int ny = args.ny;
    const std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);

    auto runtime = dcl::Runtime::create(argc, argv);
    runtime.discover_devices({
        args.device == DeviceFilter::cpu ? dcl::DeviceKind::cpu :
        args.device == DeviceFilter::gpu ? dcl::DeviceKind::gpu :
                                           dcl::DeviceKind::all,
        0
    });

    print_devices_all_ranks_dcl(runtime, rank, world);
    print_exec_characteristics_rank0(args, rank);

    std::vector<float> a(total, 0.0f), b(total, 0.0f);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            a[static_cast<std::size_t>(y) * nx + x] = 0.001f * static_cast<float>((x + 3 * y) % 1000);
        }
    }
    b = a;

    auto kernel = runtime.create_kernel({args.kernel_file, heavy ? "lu_update_2d_heavy_flat" : "ocean_jacobi_2d_flat", ""});
    runtime.set_partition({total, 1, sizeof(float), static_cast<std::size_t>(nx)});

    auto f_a = runtime.create_field({heavy ? "lu_a" : "ocean_a", total, 1, sizeof(float), dcl::BufferUsage::read_write, a.data(), dcl::RedistributionDependency::proportional});
    auto f_b = runtime.create_field({heavy ? "lu_b" : "ocean_b", total, 1, sizeof(float), dcl::BufferUsage::read_write, b.data(), dcl::RedistributionDependency::proportional});

    auto bind_ab = runtime.bind(kernel).arg(0, f_a).arg(1, f_b).arg(2, nx).arg(3, ny);
    if (heavy) bind_ab = bind_ab.arg(4, args.flop_repeat);
    auto built_ab = bind_ab.build();

    auto bind_ba = runtime.bind(kernel).arg(0, f_b).arg(1, f_a).arg(2, nx).arg(3, ny);
    if (heavy) bind_ba = bind_ba.arg(4, args.flop_repeat);
    auto built_ba = bind_ba.build();

    auto step_ab = runtime.step(heavy ? "lu_ab" : "ocean_ab")
        .invoke(built_ab, dcl::LaunchGeometry{0, total, std::optional<std::size_t>()})
        .with_halo_exchange(dcl::HaloSpec{static_cast<std::size_t>(nx), std::vector<dcl::FieldHandle>{f_a}})
        .with_balance(dcl::AutoBalancePolicy{dcl::BalanceMode::off, 0})
        .synchronize_at_end(false)
        .build();

    auto step_ba = runtime.step(heavy ? "lu_ba" : "ocean_ba")
        .invoke(built_ba, dcl::LaunchGeometry{0, total, std::optional<std::size_t>()})
        .with_halo_exchange(dcl::HaloSpec{static_cast<std::size_t>(nx), std::vector<dcl::FieldHandle>{f_b}})
        .with_balance(dcl::AutoBalancePolicy{dcl::BalanceMode::off, 0})
        .synchronize_at_end(false)
        .build();

   // runtime.synchronize(true);
    const auto t0 = clock_t::now();
    for (int it = 0; it < args.iterations; ++it) {
        if ((it & 1) == 0) runtime.execute(step_ab);
        else runtime.execute(step_ba);
    }
    //runtime.synchronize(true);
    const auto t1 = clock_t::now();

    const bool final_is_a = (args.iterations % 2 == 0);
    std::vector<float> out(total, 0.0f);
    runtime.gather(final_is_a ? f_a : f_b, out.data(), out.size() * sizeof(float));
   // runtime.synchronize(true);

    if (runtime.rank() == 0) {
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << (heavy ? "LU DCL\n" : "OCEAN DCL\n");
        std::cout << "tempo_total_s=" << sec << "\n";
        std::cout << "tempo_medio_iter_s=" << (sec / args.iterations) << "\n";
        print_samples_rank0(out, sample_indices(out.size()), "samples", rank);
    }
}

inline void run_map_dcl(const Args& args, int argc, char** argv, int rank, int world) {
    using clock_t = std::chrono::steady_clock;
    const std::size_t n = static_cast<std::size_t>(args.n);

    auto runtime = dcl::Runtime::create(argc, argv);
    runtime.discover_devices({
        args.device == DeviceFilter::cpu ? dcl::DeviceKind::cpu :
        args.device == DeviceFilter::gpu ? dcl::DeviceKind::gpu :
                                           dcl::DeviceKind::all,
        0
    });

    print_devices_all_ranks_dcl(runtime, rank, world);
    print_exec_characteristics_rank0(args, rank);

    std::vector<float> in(n, 0.0f), out(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        in[i] = 0.001f * static_cast<float>(i);
    }

    auto kernel = runtime.create_kernel({args.kernel_file, "map_heavy_1d", ""});
    runtime.set_partition({n, 1, sizeof(float), 0});

    auto f_in = runtime.create_field({"map_in", n, 1, sizeof(float), dcl::BufferUsage::read_only, in.data(), dcl::RedistributionDependency::proportional});
    auto f_out = runtime.create_field({"map_out", n, 1, sizeof(float), dcl::BufferUsage::read_write, out.data(), dcl::RedistributionDependency::proportional});

    auto bind = runtime.bind(kernel)
        .arg(0, f_in)
        .arg(1, f_out)
        .arg(2, static_cast<int>(n))
        .arg(3, args.flop_repeat)
        .build();

    auto step = runtime.step("map_step")
        .invoke(bind, dcl::LaunchGeometry{0, n, std::optional<std::size_t>()})
        .with_balance(dcl::AutoBalancePolicy{dcl::BalanceMode::off, 0})
        .synchronize_at_end(false)
        .build();

    //runtime.synchronize(true);
    const auto t0 = clock_t::now();
    for (int it = 0; it < args.iterations; ++it) {
        runtime.execute(step);
    }
    //runtime.synchronize(true);
    const auto t1 = clock_t::now();

    std::vector<float> gathered(n, 0.0f);
    runtime.gather(f_out, gathered.data(), gathered.size() * sizeof(float));
    //runtime.synchronize(true);

    if (runtime.rank() == 0) {
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "MAP DCL\n";
        std::cout << "tempo_total_s=" << sec << "\n";
        std::cout << "tempo_medio_iter_s=" << (sec / args.iterations) << "\n";
        print_samples_rank0(gathered, sample_indices(gathered.size()), "samples", rank);
    }
}

inline void run_ocean_dcl(const Args& args, int argc, char** argv, int rank, int world) {
    run_stencil_dcl(args, false, argc, argv, rank, world);
}

inline void run_lu_dcl(const Args& args, int argc, char** argv, int rank, int world) {
    run_stencil_dcl(args, true, argc, argv, rank, world);
}

#endif

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    try {
        int rank = 0, world = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world);

        const Args args = parse_args(argc, argv);
        write_kernel_file(args.kernel_file);

        switch (args.impl) {
            case ImplKind::mpiocl:
                switch (args.kernel) {
                    case KernelKind::ocean: run_ocean_mpiocl_dcl_like(args, rank, world); break;
                    case KernelKind::lu:    run_lu_mpiocl_dcl_like(args, rank, world);    break;
                    case KernelKind::map:   run_map_mpiocl_dcl_like(args, rank, world);   break;
                }
                break;
            case ImplKind::dcl_impl:
#ifndef USE_DCL
                if (rank == 0) fail("DCL path requested, but this file was built without -DUSE_DCL");
#else
                switch (args.kernel) {
                    case KernelKind::ocean: run_ocean_dcl(args, argc, argv, rank, world); break;
                    case KernelKind::lu:    run_lu_dcl(args, argc, argv, rank, world);    break;
                    case KernelKind::map:   run_map_dcl(args, argc, argv, rank, world);   break;
                }
#endif
                break;
        }

        MPI_Finalize();
        return 0;
    } catch (const std::exception& e) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cerr << "rank " << rank << " error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
