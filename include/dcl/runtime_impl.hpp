#ifndef DCL_RUNTIME_IMPL_HPP
#define DCL_RUNTIME_IMPL_HPP

#include "runtime.hpp"

// #ifndef CL_TARGET_OPENCL_VERSION
// #define CL_TARGET_OPENCL_VERSION 120
// #endif
#include <CL/cl.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace dcl {

namespace detail {

static inline void check_mpi(int code, const char* what) {
    if (code == MPI_SUCCESS) return;

    char err_string[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(code, err_string, &len);

    std::ostringstream oss;
    oss << what << " failed: " << std::string(err_string, len);
    throw Error(oss.str());
}

static inline void check_cl(cl_int code, const char* what) {
    if (code == CL_SUCCESS) return;

    std::ostringstream oss;
    oss << "OpenCL failure at " << what << " code=" << code;
    throw Error(oss.str());
}

static inline std::string slurp_file(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) {
        throw Error("Could not open kernel source file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static inline DeviceKind classify_device_kind(cl_device_type type) {
    if (type & CL_DEVICE_TYPE_GPU) return DeviceKind::gpu;
    if (type & CL_DEVICE_TYPE_CPU) return DeviceKind::cpu;
    if (type & CL_DEVICE_TYPE_ACCELERATOR) return DeviceKind::accelerator;
    return DeviceKind::all;
}

static inline bool matches(DeviceKind wanted, DeviceKind actual) {
    return wanted == DeviceKind::all || wanted == actual;
}

static inline cl_mem_flags to_opencl_flags(BufferUsage usage, bool has_host_ptr) {
    cl_mem_flags flags = 0;
    switch (usage) {
        case BufferUsage::read_only:  flags |= CL_MEM_READ_ONLY;  break;
        case BufferUsage::write_only: flags |= CL_MEM_WRITE_ONLY; break;
        case BufferUsage::read_write: flags |= CL_MEM_READ_WRITE; break;
    }
    if (has_host_ptr) flags |= CL_MEM_COPY_HOST_PTR;
    return flags;
}

static inline bool intersect_1d(
    std::size_t off1,
    std::size_t len1,
    std::size_t off2,
    std::size_t len2,
    std::size_t& out_off,
    std::size_t& out_len
) {
    const std::size_t a0 = off1;
    const std::size_t a1 = off1 + len1;
    const std::size_t b0 = off2;
    const std::size_t b1 = off2 + len2;

    const std::size_t lo = std::max(a0, b0);
    const std::size_t hi = std::min(a1, b1);

    if (hi <= lo) {
        out_off = 0;
        out_len = 0;
        return false;
    }

    out_off = lo;
    out_len = hi - lo;
    return true;
}

static inline std::vector<float> compute_loads_from_times(const std::vector<double>& times) {
    std::vector<float> loads(times.size(), 0.0f);
    if (times.empty()) return loads;

    double sum_inv = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        double t = times[i];
        if (t < 1.0e-12) t = 1.0e-12;
        sum_inv += 1.0 / t;
    }

    if (sum_inv <= 0.0) {
        const float eq = 1.0f / static_cast<float>(times.size());
        std::fill(loads.begin(), loads.end(), eq);
        return loads;
    }

    for (std::size_t i = 0; i < times.size(); ++i) {
        double t = times[i];
        if (t < 1.0e-12) t = 1.0e-12;
        loads[i] = static_cast<float>((1.0 / t) / sum_inv);
    }

    return loads;
}



static inline float l2_norm_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return static_cast<float>(std::sqrt(acc));
}

struct PlatformContext {
    cl_platform_id platform{nullptr};
    cl_context context{nullptr};
    std::vector<cl_device_id> devices;
};

struct LocalDevice {
    int platform_index{-1};
    int device_index_in_platform{-1};
    int local_index{-1};
    int global_index{-1};

    cl_device_id device{nullptr};
    cl_context context{nullptr};
    cl_command_queue kernel_queue{nullptr};
    cl_command_queue transfer_queue{nullptr};

    cl_uint compute_units{0};
    DeviceKind kind{DeviceKind::all};
    std::string name;
};

struct RegisteredBuffer {
    BufferSpec spec;
    std::vector<cl_mem> replicas;
};

struct RegisteredField {
    FieldSpec spec;
    std::vector<cl_mem> replicas;
};

struct RegisteredKernel {
    KernelSpec spec;
    std::vector<cl_program> programs_per_platform;
    std::vector<cl_kernel> kernels_per_local_device;
};

} // namespace detail

using detail::PlatformContext;
using detail::LocalDevice;
using detail::RegisteredBuffer;
using detail::RegisteredField;
using detail::RegisteredKernel;
using detail::check_mpi;
using detail::check_cl;
using detail::slurp_file;
using detail::classify_device_kind;
using detail::matches;
using detail::to_opencl_flags;
using detail::intersect_1d;
using detail::compute_loads_from_times;
using detail::l2_norm_diff;

class Runtime::Impl {
public:
    Impl(int& argc, char**& argv)
        : mpi_initialized_by_runtime_(false), argc_(argc), argv_(argv) {}

    ~Impl() {
        for (std::unordered_map<int, RegisteredKernel>::iterator it = kernels_.begin(); it != kernels_.end(); ++it) {
            RegisteredKernel& rk = it->second;
            for (std::size_t i = 0; i < rk.kernels_per_local_device.size(); ++i) {
                if (rk.kernels_per_local_device[i] != nullptr) {
                    clReleaseKernel(rk.kernels_per_local_device[i]);
                }
            }
            for (std::size_t i = 0; i < rk.programs_per_platform.size(); ++i) {
                if (rk.programs_per_platform[i] != nullptr) {
                    clReleaseProgram(rk.programs_per_platform[i]);
                }
            }
        }

        for (std::unordered_map<int, RegisteredBuffer>::iterator it = buffers_.begin(); it != buffers_.end(); ++it) {
            RegisteredBuffer& rb = it->second;
            for (std::size_t i = 0; i < rb.replicas.size(); ++i) {
                if (rb.replicas[i] != nullptr) clReleaseMemObject(rb.replicas[i]);
            }
        }

        for (std::unordered_map<int, RegisteredField>::iterator it = fields_.begin(); it != fields_.end(); ++it) {
            RegisteredField& rf = it->second;
            for (std::size_t i = 0; i < rf.replicas.size(); ++i) {
                if (rf.replicas[i] != nullptr) clReleaseMemObject(rf.replicas[i]);
            }
        }

        for (std::size_t i = 0; i < local_devices_.size(); ++i) {
            if (local_devices_[i].kernel_queue != nullptr) clReleaseCommandQueue(local_devices_[i].kernel_queue);
            if (local_devices_[i].transfer_queue != nullptr) clReleaseCommandQueue(local_devices_[i].transfer_queue);
        }

        for (std::size_t i = 0; i < platforms_.size(); ++i) {
            if (platforms_[i].context != nullptr) clReleaseContext(platforms_[i].context);
        }
    }

    void initialize_mpi_from_runtime() {
        int initialized = 0;
        MPI_Initialized(&initialized);

        if (!initialized) {
            int provided = 0;
            MPI_Init_thread(&argc_, &argv_, MPI_THREAD_SERIALIZED, &provided);
            mpi_initialized_by_runtime_ = true;
        } else {
            mpi_initialized_by_runtime_ = false;
        }

        check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank_), "MPI_Comm_rank");
        check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &size_), "MPI_Comm_size");
        comm_ = MPI_COMM_WORLD;
    }

    void discover_devices(const DeviceSelection& selection) {
        clear_runtime_state();

        cl_uint n_platforms = 0;
        check_cl(clGetPlatformIDs(0, nullptr, &n_platforms), "clGetPlatformIDs(count)");

        if (n_platforms == 0) {
            all_device_counts_.assign(size_, 0);
            return;
        }

        std::vector<cl_platform_id> platform_ids(n_platforms);
        check_cl(clGetPlatformIDs(n_platforms, platform_ids.data(), nullptr), "clGetPlatformIDs(list)");

        int next_local_index = 0;
        const int max_per_rank =
            (selection.max_devices_per_rank <= 0)
                ? std::numeric_limits<int>::max()
                : selection.max_devices_per_rank;

        for (cl_uint p = 0; p < n_platforms; ++p) {
            cl_uint dev_count = 0;
            cl_int err = clGetDeviceIDs(platform_ids[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &dev_count);
            if (err == CL_DEVICE_NOT_FOUND || dev_count == 0) {
                continue;
            }
            check_cl(err, "clGetDeviceIDs(count)");

            std::vector<cl_device_id> platform_devices(dev_count);
            check_cl(
                clGetDeviceIDs(platform_ids[p], CL_DEVICE_TYPE_ALL, dev_count, platform_devices.data(), nullptr),
                "clGetDeviceIDs(list)"
            );

            std::vector<cl_device_id> selected_devices;
            for (cl_uint i = 0; i < dev_count; ++i) {
                if (next_local_index >= max_per_rank) break;

                cl_device_type dtype = 0;
                check_cl(
                    clGetDeviceInfo(platform_devices[i], CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr),
                    "clGetDeviceInfo(CL_DEVICE_TYPE)"
                );

                const DeviceKind kind = classify_device_kind(dtype);
                if (!matches(selection.kind, kind)) continue;

                selected_devices.push_back(platform_devices[i]);
                ++next_local_index;
            }

            if (selected_devices.empty()) continue;

            PlatformContext pc{};
            pc.platform = platform_ids[p];
            pc.devices = selected_devices;

            cl_context_properties props[] = {
                CL_CONTEXT_PLATFORM,
                reinterpret_cast<cl_context_properties>(platform_ids[p]),
                0
            };

            cl_int ctx_err = CL_SUCCESS;
            pc.context = clCreateContext(
                props,
                static_cast<cl_uint>(selected_devices.size()),
                selected_devices.data(),
                nullptr,
                nullptr,
                &ctx_err
            );
            check_cl(ctx_err, "clCreateContext");

            platforms_.push_back(pc);
            const int platform_index = static_cast<int>(platforms_.size() - 1);

            for (std::size_t i = 0; i < selected_devices.size(); ++i) {
                LocalDevice ld{};
                ld.platform_index = platform_index;
                ld.device_index_in_platform = static_cast<int>(i);
                ld.local_index = static_cast<int>(local_devices_.size());
                ld.device = selected_devices[i];
                ld.context = platforms_[platform_index].context;

                char name_buf[512] = {};
                cl_uint cus = 0;
                cl_device_type dtype = 0;

                check_cl(clGetDeviceInfo(ld.device, CL_DEVICE_NAME, sizeof(name_buf), name_buf, nullptr), "clGetDeviceInfo(CL_DEVICE_NAME)");
                check_cl(clGetDeviceInfo(ld.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr), "clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)");
                check_cl(clGetDeviceInfo(ld.device, CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr), "clGetDeviceInfo(CL_DEVICE_TYPE)");

                ld.name = std::string(name_buf);
                ld.compute_units = static_cast<unsigned>(cus);
                ld.kind = classify_device_kind(dtype);

                cl_int qerr = CL_SUCCESS;
#if CL_TARGET_OPENCL_VERSION >= 200
                const cl_queue_properties qprops[] = {
                    CL_QUEUE_PROPERTIES,
                    static_cast<cl_queue_properties>(CL_QUEUE_PROFILING_ENABLE),
                    0
                };

                ld.kernel_queue = clCreateCommandQueueWithProperties(ld.context, ld.device, qprops, &qerr);
                check_cl(qerr, "clCreateCommandQueueWithProperties(kernel_queue)");
                ld.transfer_queue = clCreateCommandQueueWithProperties(ld.context, ld.device, qprops, &qerr);
                check_cl(qerr, "clCreateCommandQueueWithProperties(transfer_queue)");
#else
                ld.kernel_queue = clCreateCommandQueue(ld.context, ld.device, CL_QUEUE_PROFILING_ENABLE, &qerr);
                check_cl(qerr, "clCreateCommandQueue(kernel_queue)");
                ld.transfer_queue = clCreateCommandQueue(ld.context, ld.device, CL_QUEUE_PROFILING_ENABLE, &qerr);
                check_cl(qerr, "clCreateCommandQueue(transfer_queue)");
#endif

                local_devices_.push_back(ld);

                DeviceInfo info{};
                info.rank = rank_;
                info.local_index = ld.local_index;
                info.global_index = -1;
                info.name = ld.name;
                info.kind = ld.kind;
                info.compute_units = ld.compute_units;
                devices_.push_back(info);
            }

            if (next_local_index >= max_per_rank) break;
        }

        const int local_count = static_cast<int>(devices_.size());
        all_device_counts_.assign(size_, 0);

        check_mpi(
            MPI_Allgather(&local_count, 1, MPI_INT, all_device_counts_.data(), 1, MPI_INT, comm_),
            "MPI_Allgather(device_counts)"
        );

        int global_base = 0;
        for (int r = 0; r < rank_; ++r) global_base += all_device_counts_[r];

        for (std::size_t i = 0; i < devices_.size(); ++i) {
            devices_[i].global_index = global_base + static_cast<int>(i);
        }
    }

    const std::vector<DeviceInfo>& devices() const noexcept { return devices_; }
    const std::vector<DevicePartition>& partitions() const noexcept { return partitions_; }
    int rank() const noexcept { return rank_; }
    int size() const noexcept { return size_; }
    MPI_Comm communicator() const noexcept { return comm_; }

    BufferHandle create_buffer(const BufferSpec& spec) {
        if (local_devices_.empty()) throw Error("discover_devices() must be called before create_buffer()");
        if (spec.bytes == 0) throw Error("BufferSpec.bytes must be > 0");

        BufferHandle h{next_buffer_id_++};
        RegisteredBuffer rb;
        rb.spec = spec;
        rb.replicas.resize(local_devices_.size(), nullptr);

        const cl_mem_flags flags = to_opencl_flags(spec.usage, spec.host_ptr != nullptr);

        for (std::size_t d = 0; d < local_devices_.size(); ++d) {
            cl_int err = CL_SUCCESS;
            rb.replicas[d] = clCreateBuffer(local_devices_[d].context, flags, spec.bytes, const_cast<void*>(spec.host_ptr), &err);
            check_cl(err, "clCreateBuffer(create_buffer)");
        }

        buffers_.insert(std::make_pair(h.value, rb));
        return h;
    }


    std::size_t active_local_partition_count() const {
    std::size_t count = 0;
    for (std::size_t p = 0; p < partitions_.size(); ++p) {
        const DevicePartition& dp = partitions_[p];
        if (dp.owning_rank == rank_ && dp.local_index >= 0 && dp.element_count > 0) {
            ++count;
        }
    }
    return count;
}

bool needs_halo_exchange(const HaloSpec& hs) const {
    if (hs.width_elements == 0) {
        return false;
    }
    if (hs.fields.empty()) {
        return false;
    }

    // Se só existe uma partição local ativa, não há fronteira entre dispositivos.
    if (active_local_partition_count() <= 1) {
        return false;
    }

    // Se existe mais de uma partição global, há fronteiras e o halo faz sentido.
    return partitions_.size() > 1;
}
    FieldHandle create_field(const FieldSpec& spec) {
        if (local_devices_.empty()) throw Error("discover_devices() must be called before create_field()");
        if (spec.global_elements == 0) throw Error("FieldSpec.global_elements must be > 0");
        if (spec.units_per_element == 0) throw Error("FieldSpec.units_per_element must be > 0");
        if (spec.bytes_per_unit == 0) throw Error("FieldSpec.bytes_per_unit must be > 0");

        FieldHandle h{next_field_id_++};
        RegisteredField rf;
        rf.spec = spec;
        rf.replicas.resize(local_devices_.size(), nullptr);

        const std::size_t total_bytes = spec.global_elements * spec.units_per_element * spec.bytes_per_unit;
        const cl_mem_flags flags = to_opencl_flags(spec.usage, false);

        for (std::size_t d = 0; d < local_devices_.size(); ++d) {
            cl_int err = CL_SUCCESS;
            rf.replicas[d] = clCreateBuffer(local_devices_[d].context, flags, total_bytes, nullptr, &err);
            check_cl(err, "clCreateBuffer(create_field)");
        }

        fields_.insert(std::make_pair(h.value, rf));

        if (spec.host_ptr != nullptr) write_initial_field_data(h, spec.host_ptr);

        return h;
    }

    KernelHandle create_kernel(const KernelSpec& spec) {
        if (local_devices_.empty()) throw Error("discover_devices() must be called before create_kernel()");
        if (spec.source_file.empty()) throw Error("KernelSpec.source_file is empty");
        if (spec.entry_point.empty()) throw Error("KernelSpec.entry_point is empty");

        KernelHandle h{next_kernel_id_++};
        RegisteredKernel rk;
        rk.spec = spec;
        rk.programs_per_platform.resize(platforms_.size(), nullptr);
        rk.kernels_per_local_device.resize(local_devices_.size(), nullptr);

        const std::string src = slurp_file(spec.source_file);
        const char* src_ptr = src.c_str();
        const size_t src_len = src.size();

        for (std::size_t p = 0; p < platforms_.size(); ++p) {
            cl_int err = CL_SUCCESS;
            cl_program program = clCreateProgramWithSource(platforms_[p].context, 1, &src_ptr, &src_len, &err);
            check_cl(err, "clCreateProgramWithSource");

            std::string opts = spec.build_options;
            err = clBuildProgram(
                program,
                static_cast<cl_uint>(platforms_[p].devices.size()),
                platforms_[p].devices.data(),
                opts.empty() ? nullptr : opts.c_str(),
                nullptr,
                nullptr
            );

            if (err != CL_SUCCESS) {
                std::ostringstream oss;
                oss << "clBuildProgram failed for kernel " << spec.entry_point << "\n";
                for (std::size_t i = 0; i < platforms_[p].devices.size(); ++i) {
                    size_t log_size = 0;
                    clGetProgramBuildInfo(program, platforms_[p].devices[i], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                    std::vector<char> log(log_size + 1, '\0');
                    clGetProgramBuildInfo(program, platforms_[p].devices[i], CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                    oss << "---- device " << i << " ----\n" << log.data() << "\n";
                }
                clReleaseProgram(program);
                throw Error(oss.str());
            }

            rk.programs_per_platform[p] = program;
        }

        for (std::size_t d = 0; d < local_devices_.size(); ++d) {
            const int p = local_devices_[d].platform_index;
            cl_int err = CL_SUCCESS;
            cl_kernel k = clCreateKernel(rk.programs_per_platform[p], spec.entry_point.c_str(), &err);
            check_cl(err, "clCreateKernel");
            rk.kernels_per_local_device[d] = k;
        }

        kernels_.insert(std::make_pair(h.value, rk));
        active_kernel_ = h.value;
        return h;
    }

    void set_partition(const PartitionSpec& spec) {
        if (spec.global_elements == 0) throw Error("PartitionSpec.global_elements must be > 0");
        if (spec.units_per_element == 0) throw Error("PartitionSpec.units_per_element must be > 0");
        if (spec.bytes_per_unit == 0) throw Error("PartitionSpec.bytes_per_unit must be > 0");

        partition_ = spec;
        rebuild_partitions_equal_like_wrapper();

        current_loads_.assign(partitions_.size(), 0.0f);
        if (!partitions_.empty()) {
            const float eq = 1.0f / static_cast<float>(partitions_.size());
            std::fill(current_loads_.begin(), current_loads_.end(), eq);
        }
    }

    void rebalance(FieldHandle target_field);
    void extract_rw_fields(const KernelBinding& binding);
    void finalize_kernel_events(std::vector<std::pair<std::size_t, cl_event>>& kernel_events, std::vector<double>& elapsed_local);
    void synchronize_all_local_devices();
    void run_full_phase(RegisteredKernel& rk, const KernelInvocation& ki, std::vector<std::pair<std::size_t, cl_event>>& kernel_events);
    void run_interior_phase(RegisteredKernel& rk, const KernelInvocation& ki, std::vector<std::pair<std::size_t, cl_event>>& kernel_events, std::size_t halo);
    void run_border_phase(RegisteredKernel& rk, const KernelInvocation& ki, std::vector<std::pair<std::size_t, cl_event>>& kernel_events, std::size_t halo);
    void exchange_halo_set(const HaloSpec& hs);
    void exchange_halos_for_field(FieldHandle fh, std::size_t halo);
    void redistribute_field_intersection(FieldHandle fh, const std::vector<DevicePartition>& old_parts, const std::vector<DevicePartition>& new_parts);
    // void mpi_reduce_bytes_bor(const unsigned char* sendbuf, unsigned char* recvbuf, std::size_t total_bytes, int root);
    // void mpi_exchange_bytes_chunked(const unsigned char* sendbuf, unsigned char* recvbuf, std::size_t total_bytes, int peer_rank,int tag_base);

void execute(const ExecutionStep& step) {
    if (step.invocations.empty()) {
        return;
    }

    const std::size_t local_parts = this->active_local_partition_count();
    const bool use_halo_pipeline = this->needs_halo_exchange(step.halo);

    for (std::size_t inv = 0; inv < step.invocations.size(); ++inv) {
        const KernelInvocation& ki = step.invocations[inv];

        std::unordered_map<int, detail::RegisteredKernel>::iterator kit =
            kernels_.find(ki.binding.kernel.value);
        if (kit == kernels_.end()) {
            throw Error("Unknown kernel in execute()");
        }

        detail::RegisteredKernel& rk = kit->second;
        std::vector<double> elapsed_local(local_devices_.size(), 0.0);
        std::vector<std::pair<std::size_t, cl_event>> kernel_events;

        // Caso 1:
        // apenas uma partição/dispositivo local ativa -> launch único completo
        if (local_parts <= 1) {
            this->run_full_phase(rk, ki, kernel_events);
            this->synchronize_all_local_devices();
            this->finalize_kernel_events(kernel_events, elapsed_local);
        }
        // Caso 2:
        // múltiplos dispositivos, mas sem halo -> uma fatia completa por dispositivo
        else if (!use_halo_pipeline) {
            this->run_full_phase(rk, ki, kernel_events);
            this->synchronize_all_local_devices();
            this->finalize_kernel_events(kernel_events, elapsed_local);
        }
        // Caso 3:
        // múltiplos dispositivos com halo -> interior + exchange + bordas
        else {
            this->run_interior_phase(rk, ki, kernel_events, step.halo.width_elements);
            this->exchange_halo_set(step.halo);
            this->synchronize_all_local_devices();
            this->finalize_kernel_events(kernel_events, elapsed_local);

            kernel_events.clear();

            this->run_border_phase(rk, ki, kernel_events, step.halo.width_elements);
            this->synchronize_all_local_devices();
            this->finalize_kernel_events(kernel_events, elapsed_local);
        }

        last_elapsed_local_ = elapsed_local;
        last_input_fields_.clear();
        last_output_fields_.clear();

        this->extract_rw_fields(ki.binding);

        if (step.balance.mode != BalanceMode::off) {
            const int every = (step.balance.interval <= 0) ? 1 : step.balance.interval;
            ++iteration_counter_;

            if ((iteration_counter_ % static_cast<std::size_t>(every)) == 0) {
                if (!last_output_fields_.empty()) {
                    this->rebalance(last_output_fields_.front());
                }
            }
        } else {
            ++iteration_counter_;
        }
    }

    if (step.synchronize_at_end) {
        this->synchronize_all_local_devices();
    }
}





inline void mpi_reduce_bytes_bor(const unsigned char* sendbuf,
                                                unsigned char* recvbuf,
                                                std::size_t total_bytes,
                                                int root) {
    std::size_t offset = 0;

    while (offset < total_bytes) {
        const std::size_t remaining = total_bytes - offset;
        const int chunk = static_cast<int>(
            std::min<std::size_t>(remaining, static_cast<std::size_t>(INT_MAX))
        );

        check_mpi(
            MPI_Reduce(
                sendbuf + offset,
                (rank_ == root && recvbuf != nullptr) ? (recvbuf + offset) : nullptr,
                chunk,
                MPI_BYTE,
                MPI_BOR,
                root,
                comm_
            ),
            "MPI_Reduce(chunked bytes)"
        );

        offset += static_cast<std::size_t>(chunk);
    }
}

inline void mpi_exchange_bytes_chunked(const unsigned char* sendbuf,
                                                      unsigned char* recvbuf,
                                                      std::size_t total_bytes,
                                                      int peer_rank,
                                                      int tag_base) {
    std::size_t offset = 0;
    int chunk_id = 0;

    while (offset < total_bytes) {
        const std::size_t remaining = total_bytes - offset;
        const int chunk = static_cast<int>(
            std::min<std::size_t>(remaining, static_cast<std::size_t>(INT_MAX))
        );

        MPI_Request reqs[2];
        int req_count = 0;

        check_mpi(
            MPI_Isend(
                sendbuf + offset,
                chunk,
                MPI_BYTE,
                peer_rank,
                tag_base + chunk_id,
                comm_,
                &reqs[req_count++]
            ),
            "MPI_Isend(chunked bytes)"
        );

        check_mpi(
            MPI_Irecv(
                recvbuf + offset,
                chunk,
                MPI_BYTE,
                peer_rank,
                tag_base + chunk_id,
                comm_,
                &reqs[req_count++]
            ),
            "MPI_Irecv(chunked bytes)"
        );

        check_mpi(
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE),
            "MPI_Waitall(chunked bytes)"
        );

        offset += static_cast<std::size_t>(chunk);
        ++chunk_id;
    }
}





void gather(FieldHandle field, void* host_dst, std::size_t bytes) {
    std::unordered_map<int, RegisteredField>::iterator fit = fields_.find(field.value);
    if (fit == fields_.end()) throw Error("Unknown field in gather()");
    if (host_dst == nullptr) throw Error("gather() host_dst is null");

    RegisteredField& rf = fit->second;
    const std::size_t total_bytes =
        rf.spec.global_elements * rf.spec.units_per_element * rf.spec.bytes_per_unit;

    if (bytes < total_bytes) throw Error("gather() destination buffer too small");

    std::vector<unsigned char> local_image(total_bytes, 0);
    std::vector<cl_event> read_events;

    for (std::size_t p = 0; p < partitions_.size(); ++p) {
        const DevicePartition& dp = partitions_[p];
        if (dp.owning_rank != rank_ || dp.local_index < 0) continue;

        const std::size_t d = static_cast<std::size_t>(dp.local_index);
        const std::size_t off =
            dp.global_offset * rf.spec.units_per_element * rf.spec.bytes_per_unit;
        const std::size_t len =
            dp.element_count * rf.spec.units_per_element * rf.spec.bytes_per_unit;

        cl_event ev = nullptr;

        check_cl(
            clEnqueueReadBuffer(
                local_devices_[d].transfer_queue,
                rf.replicas[d],
                CL_FALSE,
                off,
                len,
                local_image.data() + off,
                0,
                nullptr,
                &ev
            ),
            "clEnqueueReadBuffer(gather)"
        );

        read_events.push_back(ev);
    }

    for (std::size_t i = 0; i < read_events.size(); ++i) {
        if (read_events[i] != nullptr) {
            check_cl(
                clWaitForEvents(1, &read_events[i]),
                "clWaitForEvents(gather)"
            );
            clReleaseEvent(read_events[i]);
        }
    }

    if (rank_ == 0) {
        std::vector<unsigned char> reduced(total_bytes, 0);

        this->mpi_reduce_bytes_bor(
            local_image.data(),
            reduced.data(),
            total_bytes,
            0
        );

        std::memcpy(host_dst, reduced.data(), total_bytes);
    } else {
        this->mpi_reduce_bytes_bor(
            local_image.data(),
            nullptr,
            total_bytes,
            0
        );
    }
}

private:
    bool mpi_initialized_by_runtime_{false};
    bool mpi_is_initialized_{false};
    bool mpi_is_finalized_{false};

    void clear_runtime_state() {
        platforms_.clear();
        local_devices_.clear();
        devices_.clear();
        partitions_.clear();
        all_device_counts_.clear();
        buffers_.clear();
        fields_.clear();
        kernels_.clear();
        current_loads_.clear();
        last_elapsed_local_.clear();
        last_input_fields_.clear();
        last_output_fields_.clear();

        next_buffer_id_ = 0;
        next_field_id_ = 0;
        next_kernel_id_ = 0;
        active_kernel_ = -1;
        iteration_counter_ = 0;
    }

    int local_global_base_from_counts() const {
        int base = 0;
        for (int r = 0; r < rank_; ++r) base += all_device_counts_[r];
        return base;
    }

    int owner_rank_of_global_device(int g) const {
        int acc = 0;
        for (int r = 0; r < size_; ++r) {
            if (g < acc + all_device_counts_[r]) return r;
            acc += all_device_counts_[r];
        }
        return size_ - 1;
    }

    int local_index_of_global_device(int g) const {
        int acc = 0;
        for (int r = 0; r < size_; ++r) {
            if (g < acc + all_device_counts_[r]) return g - acc;
            acc += all_device_counts_[r];
        }
        return -1;
    }

    void rebuild_partitions_equal_like_wrapper() {
        partitions_.clear();
        if (!partition_.has_value()) return;

        int total_devices = 0;
        for (std::size_t i = 0; i < all_device_counts_.size(); ++i) total_devices += all_device_counts_[i];
        if (total_devices <= 0) return;

        const std::size_t n = partition_->global_elements;
        const std::size_t base = n / static_cast<std::size_t>(total_devices);

        std::size_t cur = 0;
        for (int g = 0; g < total_devices; ++g) {
            std::size_t len = base;
            if (g == total_devices - 1) len = n - cur;

            DevicePartition dp;
            dp.device_global_index = g;
            dp.owning_rank = owner_rank_of_global_device(g);
            dp.local_index = (dp.owning_rank == rank_) ? local_index_of_global_device(g) : -1;
            dp.global_offset = cur;
            dp.element_count = len;
            partitions_.push_back(dp);

            cur += base;
        }
    }

    std::vector<DevicePartition> partitions_from_loads(const std::vector<float>& loads) const {
        std::vector<DevicePartition> out;
        if (!partition_.has_value()) return out;
        if (loads.empty()) return out;

        std::vector<std::size_t> cuts(loads.size() + 1, 0);
        cuts[0] = 0;

        const std::size_t n = partition_->global_elements;
        for (std::size_t i = 0; i < loads.size(); ++i) {
            std::size_t end = static_cast<std::size_t>(std::llround(static_cast<double>(loads[i]) * static_cast<double>(n)));
            if (i + 1 == loads.size()) end = n;
            cuts[i + 1] = end;
        }

        for (std::size_t i = 1; i < cuts.size(); ++i) {
            if (cuts[i] < cuts[i - 1]) cuts[i] = cuts[i - 1];
            if (cuts[i] > n) cuts[i] = n;
        }
        cuts.back() = n;

        for (std::size_t g = 0; g < loads.size(); ++g) {
            DevicePartition dp;
            dp.device_global_index = static_cast<int>(g);
            dp.owning_rank = owner_rank_of_global_device(static_cast<int>(g));
            dp.local_index = (dp.owning_rank == rank_) ? local_index_of_global_device(static_cast<int>(g)) : -1;
            dp.global_offset = cuts[g];
            dp.element_count = cuts[g + 1] - cuts[g];
            out.push_back(dp);
        }

        return out;
    }

    void write_initial_field_data(FieldHandle h, const void* host_ptr) {
        std::unordered_map<int, RegisteredField>::iterator it = fields_.find(h.value);
        if (it == fields_.end()) throw Error("write_initial_field_data(): unknown field");

        RegisteredField& rf = it->second;
        const std::size_t total_bytes = rf.spec.global_elements * rf.spec.units_per_element * rf.spec.bytes_per_unit;
        std::vector<cl_event> write_events;

        for (std::size_t d = 0; d < local_devices_.size(); ++d) {
            cl_event ev = nullptr;
            check_cl(
                clEnqueueWriteBuffer(
                    local_devices_[d].transfer_queue,
                    rf.replicas[d],
                    CL_FALSE,
                    0,
                    total_bytes,
                    host_ptr,
                    0,
                    nullptr,
                    &ev
                ),
                "clEnqueueWriteBuffer(write_initial_field_data)"
            );
            write_events.push_back(ev);
        }

        for (std::size_t i = 0; i < write_events.size(); ++i) {
            if (write_events[i] != nullptr) {
                check_cl(clWaitForEvents(1, &write_events[i]), "clWaitForEvents(write_initial_field_data)");
                clReleaseEvent(write_events[i]);
            }
        }
    }

    cl_mem resolve_mem_for_local_device(const KernelArg& arg, std::size_t local_device) {
        if (const BufferHandle* bh = std::get_if<BufferHandle>(&arg)) {
            std::unordered_map<int, RegisteredBuffer>::iterator it = buffers_.find(bh->value);
            if (it == buffers_.end()) throw Error("Unknown buffer handle");
            return it->second.replicas[local_device];
        }

        if (const FieldHandle* fh = std::get_if<FieldHandle>(&arg)) {
            std::unordered_map<int, RegisteredField>::iterator it = fields_.find(fh->value);
            if (it == fields_.end()) throw Error("Unknown field handle");
            return it->second.replicas[local_device];
        }

        throw Error("KernelArg memory resolution failed");
    }

    void bind_kernel_args(std::size_t local_device, cl_kernel kernel, const KernelBinding& binding) {
        for (std::size_t i = 0; i < binding.args.size(); ++i) {
            const unsigned arg_index = binding.args[i].first;
            const KernelArg& arg = binding.args[i].second;

            if (const ScalarArg* sa = std::get_if<ScalarArg>(&arg)) {
                check_cl(clSetKernelArg(kernel, arg_index, sa->bytes.size(), sa->bytes.data()), "clSetKernelArg(scalar)");
                continue;
            }

            cl_mem mem = resolve_mem_for_local_device(arg, local_device);
            check_cl(clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &mem), "clSetKernelArg(mem)");
        }
    }

private:
    int& argc_;
    char**& argv_;
    int rank_{0};
    int size_{1};
    MPI_Comm comm_{MPI_COMM_WORLD};

    std::vector<PlatformContext> platforms_;
    std::vector<LocalDevice> local_devices_;
    std::vector<DeviceInfo> devices_;
    std::vector<int> all_device_counts_;
    std::vector<DevicePartition> partitions_;

    std::unordered_map<int, RegisteredBuffer> buffers_;
    std::unordered_map<int, RegisteredField> fields_;
    std::unordered_map<int, RegisteredKernel> kernels_;

    std::optional<PartitionSpec> partition_;
    int active_kernel_{-1};

    int next_buffer_id_{0};
    int next_field_id_{0};
    int next_kernel_id_{0};

    std::vector<float> current_loads_;
    std::vector<double> last_elapsed_local_;
    std::vector<FieldHandle> last_input_fields_;
    std::vector<FieldHandle> last_output_fields_;
    std::size_t iteration_counter_{0};
};

inline void Runtime::Impl::run_full_phase(
    RegisteredKernel& rk,
    const KernelInvocation& ki,
    std::vector<std::pair<std::size_t, cl_event>>& kernel_events
) {
    for (std::size_t p = 0; p < partitions_.size(); ++p) {
        const DevicePartition& dp = partitions_[p];
        if (dp.owning_rank != rank_ || dp.local_index < 0) continue;

        const std::size_t d = static_cast<std::size_t>(dp.local_index);
        cl_kernel kernel = rk.kernels_per_local_device[d];
        bind_kernel_args(d, kernel, ki.binding);

        const std::size_t gwo = dp.global_offset + ki.geometry.global_offset;
        const std::size_t gws = dp.element_count;

        const std::size_t* lws_ptr = nullptr;
        std::size_t lws_value = 0;
        if (ki.geometry.local_size.has_value()) {
            lws_value = *ki.geometry.local_size;
            lws_ptr = &lws_value;
        }

        cl_event kernel_ev = nullptr;
        check_cl(
            clEnqueueNDRangeKernel(local_devices_[d].kernel_queue, kernel, 1, &gwo, &gws, lws_ptr, 0, nullptr, &kernel_ev),
            "clEnqueueNDRangeKernel(full)"
        );
        kernel_events.push_back(std::make_pair(d, kernel_ev));
    }
}

inline void Runtime::Impl::run_interior_phase(
    RegisteredKernel& rk,
    const KernelInvocation& ki,
    std::vector<std::pair<std::size_t, cl_event>>& kernel_events,
    std::size_t halo
) {
    for (std::size_t p = 0; p < partitions_.size(); ++p) {
        const DevicePartition& dp = partitions_[p];
        if (dp.owning_rank != rank_ || dp.local_index < 0) continue;
        if (dp.element_count <= 2 * halo) continue;

        const std::size_t d = static_cast<std::size_t>(dp.local_index);
        cl_kernel kernel = rk.kernels_per_local_device[d];
        bind_kernel_args(d, kernel, ki.binding);

        const std::size_t gwo = dp.global_offset + halo + ki.geometry.global_offset;
        const std::size_t gws = dp.element_count - 2 * halo;

        const std::size_t* lws_ptr = nullptr;
        std::size_t lws_value = 0;
        if (ki.geometry.local_size.has_value()) {
            lws_value = *ki.geometry.local_size;
            lws_ptr = &lws_value;
        }

        cl_event kernel_ev = nullptr;
        check_cl(
            clEnqueueNDRangeKernel(local_devices_[d].kernel_queue, kernel, 1, &gwo, &gws, lws_ptr, 0, nullptr, &kernel_ev),
            "clEnqueueNDRangeKernel(interior)"
        );
        kernel_events.push_back(std::make_pair(d, kernel_ev));
    }
}

inline void Runtime::Impl::run_border_phase(
    RegisteredKernel& rk,
    const KernelInvocation& ki,
    std::vector<std::pair<std::size_t, cl_event>>& kernel_events,
    std::size_t halo
) {
    for (std::size_t p = 0; p < partitions_.size(); ++p) {
        const DevicePartition& dp = partitions_[p];
        if (dp.owning_rank != rank_ || dp.local_index < 0) continue;

        const std::size_t d = static_cast<std::size_t>(dp.local_index);
        cl_kernel kernel = rk.kernels_per_local_device[d];
        bind_kernel_args(d, kernel, ki.binding);

        const std::size_t* lws_ptr = nullptr;
        std::size_t lws_value = 0;
        if (ki.geometry.local_size.has_value()) {
            lws_value = *ki.geometry.local_size;
            lws_ptr = &lws_value;
        }

        const std::size_t left_count = std::min(halo, dp.element_count);
        const std::size_t right_count =
            (dp.element_count > halo) ? std::min(halo, dp.element_count - left_count) : 0;

        if (left_count > 0) {
            const std::size_t gwo = dp.global_offset + ki.geometry.global_offset;
            const std::size_t gws = left_count;
            cl_event kernel_ev = nullptr;
            check_cl(
                clEnqueueNDRangeKernel(local_devices_[d].kernel_queue, kernel, 1, &gwo, &gws, lws_ptr, 0, nullptr, &kernel_ev),
                "clEnqueueNDRangeKernel(border-left)"
            );
            kernel_events.push_back(std::make_pair(d, kernel_ev));
        }

        if (right_count > 0 && dp.element_count > left_count) {
            const std::size_t gwo = dp.global_offset + dp.element_count - right_count + ki.geometry.global_offset;
            const std::size_t gws = right_count;
            cl_event kernel_ev = nullptr;
            check_cl(
                clEnqueueNDRangeKernel(local_devices_[d].kernel_queue, kernel, 1, &gwo, &gws, lws_ptr, 0, nullptr, &kernel_ev),
                "clEnqueueNDRangeKernel(border-right)"
            );
            kernel_events.push_back(std::make_pair(d, kernel_ev));
        }
    }
}

inline void Runtime::Impl::finalize_kernel_events(
    std::vector<std::pair<std::size_t, cl_event>>& kernel_events,
    std::vector<double>& elapsed_local
) {
    for (std::size_t i = 0; i < kernel_events.size(); ++i) {
        const std::size_t d = kernel_events[i].first;
        cl_event ev = kernel_events[i].second;
        if (ev == nullptr) continue;

        check_cl(clWaitForEvents(1, &ev), "clWaitForEvents(kernel profiling)");

        cl_ulong t0 = 0;
        cl_ulong t1 = 0;
        const cl_int e0 = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t0, nullptr);
        const cl_int e1 = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t1, nullptr);

        if (e0 == CL_SUCCESS && e1 == CL_SUCCESS && t1 >= t0) {
            elapsed_local[d] += static_cast<double>(t1 - t0) * 1.0e-9;
        }

        clReleaseEvent(ev);
    }
    kernel_events.clear();
}

inline void Runtime::Impl::synchronize_all_local_devices() {
    std::vector<cl_event> marker_events;
    marker_events.reserve(local_devices_.size() * 2);

    for (std::size_t d = 0; d < local_devices_.size(); ++d) {
        if (local_devices_[d].kernel_queue != nullptr) {
            cl_event ev = nullptr;
#if CL_TARGET_OPENCL_VERSION >= 120
            check_cl(clEnqueueMarkerWithWaitList(local_devices_[d].kernel_queue, 0, nullptr, &ev), "clEnqueueMarkerWithWaitList(kernel queue)");
#else
            check_cl(clEnqueueMarker(local_devices_[d].kernel_queue, &ev), "clEnqueueMarker(kernel queue)");
#endif
            marker_events.push_back(ev);
        }

        if (local_devices_[d].transfer_queue != nullptr) {
            cl_event ev = nullptr;
#if CL_TARGET_OPENCL_VERSION >= 120
            check_cl(clEnqueueMarkerWithWaitList(local_devices_[d].transfer_queue, 0, nullptr, &ev), "clEnqueueMarkerWithWaitList(transfer queue)");
#else
            check_cl(clEnqueueMarker(local_devices_[d].transfer_queue, &ev), "clEnqueueMarker(transfer queue)");
#endif
            marker_events.push_back(ev);
        }
    }

    for (std::size_t i = 0; i < marker_events.size(); ++i) {
        if (marker_events[i] != nullptr) {
            check_cl(clWaitForEvents(1, &marker_events[i]), "clWaitForEvents(queue marker)");
            clReleaseEvent(marker_events[i]);
        }
    }
}

inline void Runtime::Impl::exchange_halo_set(const HaloSpec& hs) {
    if (!this->needs_halo_exchange(hs)) {
        return;
    }

    for (std::size_t i = 0; i < hs.fields.size(); ++i) {
        this->exchange_halos_for_field(hs.fields[i], hs.width_elements);
    }
}

inline void Runtime::Impl::exchange_halos_for_field(FieldHandle fh, std::size_t halo) {
    if (halo == 0 || partitions_.size() < 2) {
        return;
    }

    std::unordered_map<int, RegisteredField>::iterator it = fields_.find(fh.value);
    if (it == fields_.end()) {
        throw Error("Unknown field in halo exchange");
    }

    RegisteredField& rf = it->second;
    const std::size_t elem_bytes = rf.spec.units_per_element * rf.spec.bytes_per_unit;
    const std::size_t halo_bytes = halo * elem_bytes;

    std::vector<unsigned char> send_left(halo_bytes);
    std::vector<unsigned char> send_right(halo_bytes);
    std::vector<unsigned char> recv_left(halo_bytes);
    std::vector<unsigned char> recv_right(halo_bytes);

    for (std::size_t b = 0; b + 1 < partitions_.size(); ++b) {
        const DevicePartition& left = partitions_[b];
        const DevicePartition& right = partitions_[b + 1];

        if (left.element_count < halo || right.element_count < halo) {
            continue;
        }

        const std::size_t left_border_off =
            (left.global_offset + left.element_count - halo) * elem_bytes;
        const std::size_t left_halo_off =
            (left.global_offset + left.element_count) * elem_bytes;
        const std::size_t right_border_off =
            right.global_offset * elem_bytes;
        const std::size_t right_halo_off =
            (right.global_offset - halo) * elem_bytes;

        cl_event read_left_to_right_ev = nullptr;
        cl_event read_right_to_left_ev = nullptr;
        cl_event write_left_ev = nullptr;
        cl_event write_right_ev = nullptr;

        if (left.owning_rank == rank_) {
            const std::size_t dl = static_cast<std::size_t>(left.local_index);
            check_cl(
                clEnqueueReadBuffer(
                    local_devices_[dl].transfer_queue,
                    rf.replicas[dl],
                    CL_FALSE,
                    left_border_off,
                    halo_bytes,
                    send_right.data(),
                    0,
                    nullptr,
                    &read_left_to_right_ev
                ),
                "clEnqueueReadBuffer(halo left->right send)"
            );
        }

        if (right.owning_rank == rank_) {
            const std::size_t dr = static_cast<std::size_t>(right.local_index);
            check_cl(
                clEnqueueReadBuffer(
                    local_devices_[dr].transfer_queue,
                    rf.replicas[dr],
                    CL_FALSE,
                    right_border_off,
                    halo_bytes,
                    send_left.data(),
                    0,
                    nullptr,
                    &read_right_to_left_ev
                ),
                "clEnqueueReadBuffer(halo right->left send)"
            );
        }

        // Caso local-local
        if (left.owning_rank == rank_ && right.owning_rank == rank_) {
            const std::size_t dl = static_cast<std::size_t>(left.local_index);
            const std::size_t dr = static_cast<std::size_t>(right.local_index);

            if (read_left_to_right_ev != nullptr) {
                check_cl(
                    clWaitForEvents(1, &read_left_to_right_ev),
                    "clWaitForEvents(halo left->right read ready)"
                );
                clReleaseEvent(read_left_to_right_ev);
                read_left_to_right_ev = nullptr;

                check_cl(
                    clEnqueueWriteBuffer(
                        local_devices_[dr].transfer_queue,
                        rf.replicas[dr],
                        CL_FALSE,
                        right_halo_off,
                        halo_bytes,
                        send_right.data(),
                        0,
                        nullptr,
                        &write_right_ev
                    ),
                    "clEnqueueWriteBuffer(halo left->right local)"
                );
            }

            if (read_right_to_left_ev != nullptr) {
                check_cl(
                    clWaitForEvents(1, &read_right_to_left_ev),
                    "clWaitForEvents(halo right->left read ready)"
                );
                clReleaseEvent(read_right_to_left_ev);
                read_right_to_left_ev = nullptr;

                check_cl(
                    clEnqueueWriteBuffer(
                        local_devices_[dl].transfer_queue,
                        rf.replicas[dl],
                        CL_FALSE,
                        left_halo_off,
                        halo_bytes,
                        send_left.data(),
                        0,
                        nullptr,
                        &write_left_ev
                    ),
                    "clEnqueueWriteBuffer(halo right->left local)"
                );
            }

            if (write_right_ev != nullptr) {
                check_cl(
                    clWaitForEvents(1, &write_right_ev),
                    "clWaitForEvents(halo left->right write done)"
                );
                clReleaseEvent(write_right_ev);
                write_right_ev = nullptr;
            }

            if (write_left_ev != nullptr) {
                check_cl(
                    clWaitForEvents(1, &write_left_ev),
                    "clWaitForEvents(halo right->left write done)"
                );
                clReleaseEvent(write_left_ev);
                write_left_ev = nullptr;
            }

            continue;
        }

        // Caso remoto
        if (left.owning_rank == rank_ && read_left_to_right_ev != nullptr) {
            check_cl(
                clWaitForEvents(1, &read_left_to_right_ev),
                "clWaitForEvents(halo left->right send ready)"
            );
            clReleaseEvent(read_left_to_right_ev);
            read_left_to_right_ev = nullptr;
        }

        if (right.owning_rank == rank_ && read_right_to_left_ev != nullptr) {
            check_cl(
                clWaitForEvents(1, &read_right_to_left_ev),
                "clWaitForEvents(halo right->left send ready)"
            );
            clReleaseEvent(read_right_to_left_ev);
            read_right_to_left_ev = nullptr;
        }

        // left -> right
        if (left.owning_rank == rank_) {
            const int tag_base = 700000 + static_cast<int>(b) * 2000 + fh.value * 10;
            this->mpi_exchange_bytes_chunked(
                send_right.data(),
                recv_right.data(),
                halo_bytes,
                right.owning_rank,
                tag_base
            );
        } else if (right.owning_rank == rank_) {
            const int tag_base = 700000 + static_cast<int>(b) * 2000 + fh.value * 10;
            this->mpi_exchange_bytes_chunked(
                send_left.data(),   // ignorado no rank receptor
                recv_left.data(),
                halo_bytes,
                left.owning_rank,
                tag_base
            );
        }

        // right -> left
        if (right.owning_rank == rank_) {
            const int tag_base = 800000 + static_cast<int>(b) * 2000 + fh.value * 10;
            this->mpi_exchange_bytes_chunked(
                send_left.data(),
                recv_left.data(),
                halo_bytes,
                left.owning_rank,
                tag_base
            );
        } else if (left.owning_rank == rank_) {
            const int tag_base = 800000 + static_cast<int>(b) * 2000 + fh.value * 10;
            this->mpi_exchange_bytes_chunked(
                send_right.data(),  // ignorado no rank receptor
                recv_right.data(),
                halo_bytes,
                right.owning_rank,
                tag_base
            );
        }

        if (right.owning_rank == rank_) {
            const std::size_t dr = static_cast<std::size_t>(right.local_index);
            check_cl(
                clEnqueueWriteBuffer(
                    local_devices_[dr].transfer_queue,
                    rf.replicas[dr],
                    CL_FALSE,
                    right_halo_off,
                    halo_bytes,
                    recv_left.data(),
                    0,
                    nullptr,
                    &write_right_ev
                ),
                "clEnqueueWriteBuffer(halo left->right remote)"
            );
        }

        if (left.owning_rank == rank_) {
            const std::size_t dl = static_cast<std::size_t>(left.local_index);
            check_cl(
                clEnqueueWriteBuffer(
                    local_devices_[dl].transfer_queue,
                    rf.replicas[dl],
                    CL_FALSE,
                    left_halo_off,
                    halo_bytes,
                    recv_right.data(),
                    0,
                    nullptr,
                    &write_left_ev
                ),
                "clEnqueueWriteBuffer(halo right->left remote)"
            );
        }

        if (write_right_ev != nullptr) {
            check_cl(
                clWaitForEvents(1, &write_right_ev),
                "clWaitForEvents(halo left->right remote write done)"
            );
            clReleaseEvent(write_right_ev);
            write_right_ev = nullptr;
        }

        if (write_left_ev != nullptr) {
            check_cl(
                clWaitForEvents(1, &write_left_ev),
                "clWaitForEvents(halo right->left remote write done)"
            );
            clReleaseEvent(write_left_ev);
            write_left_ev = nullptr;
        }
    }
}

inline void Runtime::Impl::extract_rw_fields(const KernelBinding& binding) {
    for (std::size_t i = 0; i < binding.args.size(); ++i) {
        const unsigned idx = binding.args[i].first;
        const KernelArg& arg = binding.args[i].second;
        if (const FieldHandle* fh = std::get_if<FieldHandle>(&arg)) {
            if (idx == 0) last_output_fields_.push_back(*fh);
            if (idx == 1) last_input_fields_.push_back(*fh);
        }
    }
}

inline void Runtime::Impl::redistribute_field_intersection(
    FieldHandle fh,
    const std::vector<DevicePartition>& old_parts,
    const std::vector<DevicePartition>& new_parts
) {
    std::unordered_map<int, RegisteredField>::iterator fit = fields_.find(fh.value);
    if (fit == fields_.end()) throw Error("redistribute_field_intersection(): unknown field");

    RegisteredField& rf = fit->second;
    const std::size_t elem_bytes = rf.spec.units_per_element * rf.spec.bytes_per_unit;
    const std::size_t total_bytes = rf.spec.global_elements * elem_bytes;
    std::vector<unsigned char> tmp(total_bytes);

    for (std::size_t src = 0; src < old_parts.size(); ++src) {
        for (std::size_t dst = 0; dst < new_parts.size(); ++dst) {
            std::size_t inter_off = 0;
            std::size_t inter_len = 0;
            if (!intersect_1d(old_parts[src].global_offset, old_parts[src].element_count,
                              new_parts[dst].global_offset, new_parts[dst].element_count,
                              inter_off, inter_len)) {
                continue;
            }
            if (inter_len == 0) continue;

            const std::size_t byte_off = inter_off * elem_bytes;
            const std::size_t byte_len = inter_len * elem_bytes;
            const int rank_src = old_parts[src].owning_rank;
            const int rank_dst = new_parts[dst].owning_rank;

            cl_event read_ev = nullptr;
            if (rank_src == rank_ && old_parts[src].local_index >= 0) {
                const std::size_t dsrc = static_cast<std::size_t>(old_parts[src].local_index);
                check_cl(
                    clEnqueueReadBuffer(local_devices_[dsrc].transfer_queue, rf.replicas[dsrc], CL_FALSE,
                                        byte_off, byte_len, tmp.data() + byte_off, 0, nullptr, &read_ev),
                    "clEnqueueReadBuffer(redistribute read)"
                );
                check_cl(clWaitForEvents(1, &read_ev), "clWaitForEvents(redistribute read)");
                clReleaseEvent(read_ev);
            }

            if (rank_src == rank_ && rank_dst == rank_) {
                if (new_parts[dst].local_index >= 0 && old_parts[src].local_index != new_parts[dst].local_index) {
                    const std::size_t ddst = static_cast<std::size_t>(new_parts[dst].local_index);
                    check_cl(
                        clEnqueueWriteBuffer(local_devices_[ddst].transfer_queue, rf.replicas[ddst], CL_FALSE,
                                             byte_off, byte_len, tmp.data() + byte_off, 0, nullptr, nullptr),
                        "clEnqueueWriteBuffer(redistribute local)"
                    );
                }
            } else {
                const int tag = 9000 + static_cast<int>(src * old_parts.size() + dst) + fh.value * 10000;
                MPI_Request reqs[2];
                int req_count = 0;

                if (rank_dst == rank_ && new_parts[dst].local_index >= 0) {
                    check_mpi(
                        MPI_Irecv(tmp.data() + byte_off, static_cast<int>(byte_len), MPI_BYTE, rank_src, tag, comm_, &reqs[req_count++]),
                        "MPI_Irecv(redistribute)"
                    );
                }

                if (rank_src == rank_) {
                    check_mpi(
                        MPI_Isend(tmp.data() + byte_off, static_cast<int>(byte_len), MPI_BYTE, rank_dst, tag, comm_, &reqs[req_count++]),
                        "MPI_Isend(redistribute)"
                    );
                }

                if (req_count > 0) {
                    check_mpi(MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE), "MPI_Waitall(redistribute)");
                }

                if (rank_dst == rank_ && new_parts[dst].local_index >= 0) {
                    const std::size_t ddst = static_cast<std::size_t>(new_parts[dst].local_index);
                    check_cl(
                        clEnqueueWriteBuffer(local_devices_[ddst].transfer_queue, rf.replicas[ddst], CL_FALSE,
                                             byte_off, byte_len, tmp.data() + byte_off, 0, nullptr, nullptr),
                        "clEnqueueWriteBuffer(redistribute remote)"
                    );
                }
            }
        }
    }

    synchronize_all_local_devices();
}

inline void Runtime::Impl::rebalance(FieldHandle target_field) {
    if (!partition_.has_value()) return;
    if (partitions_.empty()) return;

    std::unordered_map<int, RegisteredField>::iterator fit = fields_.find(target_field.value);
    if (fit == fields_.end()) throw Error("rebalance() received unknown field");

    std::vector<double> local_times(partitions_.size(), 0.0);
    for (std::size_t i = 0; i < partitions_.size(); ++i) {
        if (partitions_[i].owning_rank == rank_ && partitions_[i].local_index >= 0) {
            const int li = partitions_[i].local_index;
            if (li >= 0 && static_cast<std::size_t>(li) < last_elapsed_local_.size()) {
                local_times[i] = last_elapsed_local_[li];
            }
        }
    }

    std::vector<double> global_times(partitions_.size(), 0.0);

    if (global_times.size() > static_cast<std::size_t>(INT_MAX)) {
        throw Error("rebalance() too many partitions for MPI_Allreduce");
    }

    check_mpi(
        MPI_Allreduce(
            local_times.data(),
            global_times.data(),
            static_cast<int>(global_times.size()),
            MPI_DOUBLE,
            MPI_SUM,
            comm_
        ),
        "MPI_Allreduce(rebalance times)"
    );

    const std::vector<float> new_loads = compute_loads_from_times(global_times);
    if (new_loads.empty()) return;

    if (!current_loads_.empty()) {
        const float n = l2_norm_diff(current_loads_, new_loads);
        if (n <= 0.000025f) return;
    }

    std::vector<DevicePartition> old_parts = partitions_;
    std::vector<DevicePartition> new_parts = partitions_from_loads(new_loads);

    redistribute_field_intersection(target_field, old_parts, new_parts);

    for (std::size_t i = 0; i < last_input_fields_.size(); ++i) {
        if (last_input_fields_[i].value != target_field.value) {
            redistribute_field_intersection(last_input_fields_[i], old_parts, new_parts);
        }
    }

    for (std::size_t i = 0; i < last_output_fields_.size(); ++i) {
        if (last_output_fields_[i].value != target_field.value) {
            redistribute_field_intersection(last_output_fields_[i], old_parts, new_parts);
        }
    }

    partitions_ = new_parts;
    current_loads_ = new_loads;
}

} // namespace dcl

#endif
