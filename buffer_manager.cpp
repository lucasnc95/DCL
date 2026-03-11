#include "data/buffer_manager.hpp"
#include <algorithm>
#include <iostream>


namespace dcl {

int BufferManager::create_buffer(size_t total_elements, size_t elem_size, cl_mem_flags flags, DeviceManager& dev_mgr) {
    int local_count = dev_mgr.get_local_count();
    if (local_count <= 0) {
        std::cerr << "Erro: Tentativa de criar buffer sem dispositivos inicializados." << std::endl;
        return -1;
    }

    int global_id = m_next_id++;
    m_metadata[global_id] = {total_elements, elem_size, flags};

    size_t offset_accum = 0;
    size_t chunk = total_elements / local_count;

    for (int i = 0; i < local_count; ++i) {
        Device& dev = dev_mgr.get_device(i);
        size_t local_len = (i == local_count - 1) ? (total_elements - offset_accum) : chunk;
        
        cl_int err;
        cl_mem mem = clCreateBuffer(dev.context, flags, local_len * elem_size, nullptr, &err);
        dev.memoryObjects[global_id] = mem;

        // Inicializa a partição que será usada no Comms e no Enqueue
        m_partitions[i] = { i, offset_accum, local_len };
        offset_accum += local_len;
    }
    return global_id;
}

void BufferManager::write_global(int global_id, const void* data, size_t offset, size_t size, DeviceManager& dev_mgr) {
    auto& meta = m_metadata[global_id];
    const char* byte_data = static_cast<const char*>(data);

    for (int i = 0; i < dev_mgr.get_local_count(); ++i) {
        auto& part = m_partitions[i];
        
        // Lógica de interseção (Ref: WriteObject original)
        size_t start = std::max(offset, part.offset);
        size_t end = std::min(offset + size, part.offset + part.length);

        if (start < end) {
            size_t len_to_write = end - start;
            size_t host_offset = (start - offset) * meta.element_size;
            size_t device_offset = (start - part.offset) * meta.element_size;

            clEnqueueWriteBuffer(dev_mgr.get_device(i).dataQueue, 
                                dev_mgr.get_device(i).memoryObjects[global_id], 
                                CL_FALSE, device_offset, len_to_write * meta.element_size, 
                                byte_data + host_offset, 0, nullptr, nullptr);
        }
    }
}


void BufferManager::gather_global(int global_id, void* target_host_ptr, DeviceManager& dev_mgr) {
    auto& meta = m_metadata[global_id];
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 1. Calcula a carga local deste rank (soma de todos os devices locais)
    size_t local_elements = 0;
    for (int i = 0; i < dev_mgr.get_local_count(); ++i) {
        local_elements += m_partitions[i].length;
    }
    
    // Converter para bytes para o MPI
    int local_bytes = static_cast<int>(local_elements * meta.element_size);

    // 2. Buffer temporário para consolidar os dispositivos do nó
    std::vector<char> local_node_buffer(local_bytes);
    size_t current_offset = 0;

    for (int i = 0; i < dev_mgr.get_local_count(); ++i) {
        Device& dev = dev_mgr.get_device(i);
        size_t part_bytes = m_partitions[i].length * meta.element_size;
        
        // Leitura síncrona (CL_TRUE) para garantir que o buffer está pronto para o MPI
        cl_int err = clEnqueueReadBuffer(dev.dataQueue, dev.memoryObjects[global_id], CL_TRUE, 
                                        0, part_bytes, local_node_buffer.data() + current_offset, 
                                        0, nullptr, nullptr);
        
        if (err != CL_SUCCESS) {
            std::cerr << "[Rank " << rank << "] Erro OpenCL Read: " << err << std::endl;
        }
        current_offset += part_bytes;
    }

    // 3. Preparação do MPI_Allgatherv
    std::vector<int> recv_counts(world_size);
    std::vector<int> displacements(world_size);

    // Todos os ranks trocam suas contagens de bytes
    MPI_Allgather(&local_bytes, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    displacements[0] = 0;
    for (int i = 1; i < world_size; ++i) {
        displacements[i] = displacements[i - 1] + recv_counts[i - 1];
    }

    // 4. Comunicação Coletiva: Consolida no target_host_ptr (h_output)
    // Usamos MPI_BYTE para evitar problemas de tipos complexos
    MPI_Allgatherv(local_node_buffer.data(), local_bytes, MPI_BYTE,
                   target_host_ptr, recv_counts.data(), displacements.data(), 
                   MPI_BYTE, MPI_COMM_WORLD);
                   
    // Sincronização opcional para garantir consistência antes de retornar ao main
    MPI_Barrier(MPI_COMM_WORLD);
}


}