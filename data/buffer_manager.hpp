#ifndef DCL_BUFFER_MANAGER_HPP
#define DCL_BUFFER_MANAGER_HPP

#include <unordered_map>
#include "../core/device_manager.hpp"
#include "box.hpp"
#include <CL/cl.h>
#include <mpi.h>

namespace dcl {

struct BufferMetadata {
    size_t total_elements;
    size_t element_size;
    cl_mem_flags flags;
};

class BufferManager {
public:
    // Cria buffers e define as partições iniciais (Ref: AllocateMemoryObject)
    int create_buffer(size_t total_elements, size_t elem_size, cl_mem_flags flags, DeviceManager& dev_mgr);

    // Métodos para movimentação global (Ref: WriteObject / GatherResults)
    void write_global(int global_id, const void* data, size_t offset, size_t size, DeviceManager& dev_mgr);
    void gather_global(int global_id, void* target_host_ptr, DeviceManager& dev_mgr);

    // Ajuste de fatias para o balanceamento de carga
    void resize_partition(int local_device_idx, size_t new_offset, size_t new_length);
    
    Partition& get_partition(int local_idx) { return m_partitions[local_idx]; }

private:
    int m_next_id = 0;
    std::unordered_map<int, BufferMetadata> m_metadata;
    std::unordered_map<int, Partition> m_partitions; // local_idx -> Partition
};

} // namespace dcl
#endif