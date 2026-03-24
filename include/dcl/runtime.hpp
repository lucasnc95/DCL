#ifndef DCL_RUNTIME_HPP
#define DCL_RUNTIME_HPP

#include "types.hpp"

#include <memory>
#include <string>

namespace dcl {

class Runtime;

class KernelBindingBuilder {
public:
    KernelBindingBuilder(Runtime& runtime, KernelHandle kernel);

    KernelBindingBuilder& arg(unsigned index, BufferHandle handle);
    KernelBindingBuilder& arg(unsigned index, FieldHandle handle);
    KernelBindingBuilder& arg(unsigned index, ScalarArg scalar);

    KernelBinding build() const;

private:
    Runtime* runtime_{nullptr};
    KernelBinding binding_;
};

class StepBuilder {
public:
    StepBuilder(Runtime& runtime, std::string name);

    StepBuilder& invoke(const KernelBinding& binding, const LaunchGeometry& geometry);
    StepBuilder& with_halo_exchange(const HaloSpec& halo);
    StepBuilder& with_balance(const AutoBalancePolicy& policy);
    StepBuilder& synchronize_at_end(bool value);

    ExecutionStep build() const;

private:
    Runtime* runtime_{nullptr};
    ExecutionStep step_;
};

class Runtime {
public:
    static Runtime create(int& argc, char**& argv);

    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    void discover_devices(const DeviceSelection& selection);

    const std::vector<DeviceInfo>& devices() const noexcept;
    const std::vector<DevicePartition>& partitions() const noexcept;

    int rank() const noexcept;
    int size() const noexcept;
    MPI_Comm communicator() const noexcept;

    BufferHandle create_buffer(const BufferSpec& spec);
    FieldHandle create_field(const FieldSpec& spec);
    KernelHandle create_kernel(const KernelSpec& spec);

    KernelBindingBuilder bind(KernelHandle kernel);
    StepBuilder step(std::string name);

    void set_partition(const PartitionSpec& spec);
    void execute(const ExecutionStep& step);
    void rebalance(FieldHandle target_field);
    void gather(FieldHandle field, void* host_dst, std::size_t bytes);

private:
    class Impl;
    explicit Runtime(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;

    friend class KernelBindingBuilder;
    friend class StepBuilder;
};

} // namespace dcl

#endif