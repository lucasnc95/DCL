#ifndef DCL_TYPES_HPP
#define DCL_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <mpi.h>

namespace dcl {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

enum class DeviceKind {
    all,
    cpu,
    gpu,
    accelerator
};

enum class BufferUsage {
    read_only,
    write_only,
    read_write
};

enum class BalanceMode {
    off,
    on
};

struct BufferHandle {
    int value{-1};
};

struct FieldHandle {
    int value{-1};
};

struct KernelHandle {
    int value{-1};
};

struct ScalarArg {
    std::vector<unsigned char> bytes;

    ScalarArg() = default;

    template <typename T,
              typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
    ScalarArg(const T& value) : bytes(sizeof(T)) {
        std::memcpy(bytes.data(), &value, sizeof(T));
    }
};

using KernelArg = std::variant<BufferHandle, FieldHandle, ScalarArg>;

struct DeviceSelection {
    DeviceKind kind{DeviceKind::all};
    int max_devices_per_rank{0};
};

struct DeviceInfo {
    int rank{0};
    int local_index{-1};
    int global_index{-1};
    std::string name;
    DeviceKind kind{DeviceKind::all};
    unsigned int compute_units{0};
};

struct DevicePartition {
    int device_global_index{-1};
    int owning_rank{-1};
    int local_index{-1};
    std::size_t global_offset{0};
    std::size_t element_count{0};
};

struct BufferSpec {
    std::string name;
    std::size_t bytes{0};
    BufferUsage usage{BufferUsage::read_write};
    const void* host_ptr{nullptr};
};

struct FieldSpec {
    std::string name;
    std::size_t global_elements{0};
    std::size_t units_per_element{0};
    std::size_t bytes_per_unit{0};
    BufferUsage usage{BufferUsage::read_write};
    const void* host_ptr{nullptr};
};

struct KernelSpec {
    std::string source_file;
    std::string entry_point;
    std::string build_options;
};

struct PartitionSpec {
    std::size_t global_elements{0};
    std::size_t units_per_element{0};
    std::size_t bytes_per_unit{0};
    std::size_t halo_width_elements{0};
};

struct LaunchGeometry {
    std::size_t global_offset{0};
    std::size_t global_size{0};
    std::optional<std::size_t> local_size{};
};

struct KernelBinding {
    KernelHandle kernel;
    std::vector<std::pair<unsigned, KernelArg>> args;
};

struct KernelInvocation {
    KernelBinding binding;
    LaunchGeometry geometry;
};

struct HaloSpec {
    std::size_t width_elements{0};
    std::vector<FieldHandle> fields;
};

struct AutoBalancePolicy {
    BalanceMode mode{BalanceMode::off};
    int interval{0};
};

struct ExecutionStep {
    std::string name;
    std::vector<KernelInvocation> invocations;
    HaloSpec halo{};
    AutoBalancePolicy balance{};
    bool synchronize_at_end{true};
};

} // namespace dcl

#endif