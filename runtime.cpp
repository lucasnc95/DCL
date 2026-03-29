#include "include/dcl/runtime_impl.hpp"

namespace dcl {

Runtime Runtime::create(int& argc, char**& argv) {
    std::unique_ptr<Impl> impl(new Impl(argc, argv));
    impl->initialize_mpi_from_runtime();
    return Runtime(std::move(impl));
}

Runtime::Runtime(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;
Runtime::~Runtime() = default;

void Runtime::discover_devices(const DeviceSelection& selection) {
    impl_->discover_devices(selection);
}

const std::vector<DeviceInfo>& Runtime::devices() const noexcept {
    return impl_->devices();
}

int Runtime::rank() const noexcept {
    return impl_->rank();
}

int Runtime::size() const noexcept {
    return impl_->size();
}

MPI_Comm Runtime::communicator() const noexcept {
    return impl_->communicator();
}

BufferHandle Runtime::create_buffer(const BufferSpec& spec) {
    return impl_->create_buffer(spec);
}

FieldHandle Runtime::create_field(const FieldSpec& spec) {
    return impl_->create_field(spec);
}

KernelHandle Runtime::create_kernel(const KernelSpec& spec) {
    return impl_->create_kernel(spec);
}

KernelBindingBuilder Runtime::bind(KernelHandle kernel) {
    return KernelBindingBuilder(*this, kernel);
}

StepBuilder Runtime::step(std::string name) {
    return StepBuilder(*this, std::move(name));
}

void Runtime::set_partition(const PartitionSpec& spec) {
    impl_->set_partition(spec);
}

const std::vector<DevicePartition>& Runtime::partitions() const noexcept {
    return impl_->partitions();
}

void Runtime::execute(const ExecutionStep& step) {
    impl_->execute(step);
}

void Runtime::rebalance(FieldHandle target_field) {
    impl_->rebalance(target_field);
}

void Runtime::gather(FieldHandle field, void* host_dst, std::size_t bytes) {
    impl_->gather(field, host_dst, bytes);
}

KernelBindingBuilder::KernelBindingBuilder(Runtime& runtime, KernelHandle kernel)
    : runtime_(&runtime) {
    binding_.kernel = kernel;
}

KernelBindingBuilder& KernelBindingBuilder::arg(unsigned index, BufferHandle handle) {
    binding_.args.push_back(std::make_pair(index, KernelArg(handle)));
    return *this;
}

KernelBindingBuilder& KernelBindingBuilder::arg(unsigned index, FieldHandle handle) {
    binding_.args.push_back(std::make_pair(index, KernelArg(handle)));
    return *this;
}

KernelBindingBuilder& KernelBindingBuilder::arg(unsigned index, ScalarArg scalar) {
    binding_.args.push_back(std::make_pair(index, KernelArg(std::move(scalar))));
    return *this;
}

KernelBinding KernelBindingBuilder::build() const {
    return binding_;
}

StepBuilder::StepBuilder(Runtime& runtime, std::string name)
    : runtime_(&runtime) {
    step_.name = std::move(name);
}

StepBuilder& StepBuilder::invoke(const KernelBinding& binding, const LaunchGeometry& geometry) {
    KernelInvocation ki;
    ki.binding = binding;
    ki.geometry = geometry;
    step_.invocations.push_back(ki);
    return *this;
}

StepBuilder& StepBuilder::with_halo_exchange(const HaloSpec& halo) {
    step_.halo = halo;
    return *this;
}

StepBuilder& StepBuilder::with_balance(const AutoBalancePolicy& policy) {
    step_.balance = policy;
    return *this;
}

StepBuilder& StepBuilder::synchronize_at_end(bool value) {
    step_.synchronize_at_end = value;
    return *this;
}

ExecutionStep StepBuilder::build() const {
    return step_;
}

} // namespace dcl