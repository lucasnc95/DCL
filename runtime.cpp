#include "runtime/runtime.hpp"
#include <mpi.h>
#include <iostream>

namespace dcl {


struct ProfilingData {
    int device_id;       // ID do dispositivo que executou
    int kernel_id;       // ID do kernel (para o log)
    size_t global_size;  // Tamanho da fatia processada
    cl_event event;      // O evento associado para consulta de timing
    int rank;            // Rank MPI para identificar o nó no log
};

void CL_CALLBACK execution_complete_callback(cl_event event, cl_int status, void* user_data) {
    dcl::ProfilingData* data = static_cast<dcl::ProfilingData*>(user_data);

    cl_ulong start = 0, end = 0;
    cl_int err_start = clGetEventProfilingInfo(data->event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    cl_int err_end = clGetEventProfilingInfo(data->event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);

    if (err_start == CL_SUCCESS && err_end == CL_SUCCESS) {
        double elapsed_ms = (double)(end - start) * 1e-6;
        
        // Log formatado com precisão
        printf("[DCL][Rank %d][Dev %d] Kernel finalizado. Carga: %zu | Tempo: %.6f ms\n", 
               data->rank, data->device_id, data->global_size, elapsed_ms);
    } else {
        // Se der erro aqui, a flag de profiling provavelmente está desligada na fila
        printf("[DCL][Rank %d][Dev %d] Erro ao ler profiling: %d\n", data->rank, data->device_id, err_start);
    }

    clReleaseEvent(data->event);
    delete data;
}



Runtime::Runtime(int &argc, char** &argv) 
    : m_dev_mgr(std::make_unique<DeviceManager>()),
      m_kernel_mgr(std::make_unique<KernelManager>()),
      m_buf_mgr(std::make_unique<BufferManager>()) 
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
}

Runtime::~Runtime() {
    MPI_Finalize();
}

void Runtime::init_devices(DeviceTag tag, int max_devs) {
    m_dev_mgr->initialize(tag, max_devs);
}

void Runtime::create_kernel(const std::string& file, const std::string& name) {
    m_current_kernel_id = m_kernel_mgr->build_program(*m_dev_mgr, file, name);
}

int Runtime::create_buffer(size_t total_elements, cl_mem_flags flags) {
    // Definimos 4 bytes como padrão (float) para este teste
    return m_buf_mgr->create_buffer(total_elements, 4, flags, *m_dev_mgr);
}

void Runtime::write_buffer(int global_id, const void* data, size_t offset, size_t size) {
    m_buf_mgr->write_global(global_id, data, offset, size, *m_dev_mgr);
}

void Runtime::set_arg(int arg_idx, int global_buffer_id) {
    for (int i = 0; i < m_dev_mgr->get_local_count(); ++i) {
        Device& dev = m_dev_mgr->get_device(i);
        
        // 1. Verificar se o kernel existe para este dispositivo
        if (dev.kernels.find(m_current_kernel_id) == dev.kernels.end()) {
            std::cerr << "Erro: Kernel ID " << m_current_kernel_id << " nao encontrado no dev " << i << std::endl;
            continue;
        }

        cl_kernel k = dev.kernels[m_current_kernel_id];
        cl_mem mem_obj = dev.memoryObjects[global_buffer_id];

        // 2. Passar o endereço do objeto cl_mem
        cl_int err = clSetKernelArg(k, arg_idx, sizeof(cl_mem), &mem_obj);
        
        if (err != CL_SUCCESS) {
            std::cerr << "Erro ao setar buffer (arg " << arg_idx << ") no dev " << i << ": " << err << std::endl;
        }
    }
}

void Runtime::set_arg(int arg_idx, void* scalar_data, size_t size) {
    for (int i = 0; i < m_dev_mgr->get_local_count(); ++i) {
        Device& dev = m_dev_mgr->get_device(i);
        cl_kernel k = dev.kernels[m_current_kernel_id];
        
        // Para escalares, passamos o ponteiro para o dado e o tamanho (ex: sizeof(int))
        cl_int err = clSetKernelArg(k, arg_idx, size, scalar_data);
        
        if (err != CL_SUCCESS) {
            std::cerr << "Erro ao setar escalar (arg " << arg_idx << ") no dev " << i << ": " << err << std::endl;
        }
    }
}

void Runtime::enqueue_kernel(const DataDependency& dep) {
    // Lógica de sincronização de bordas (Halos) permanece síncrona por segurança por enquanto
    if (dep.mode == AccessMode::NEIGHBORHOOD) {
        for (int i = 0; i < m_dev_mgr->get_local_count(); ++i) {
            m_halo_mgr->sync_halos_transparent(dep.buffer_id, dep.halo_radius, 
                                               m_buf_mgr->get_partition(i), *m_dev_mgr);
        }
    }

for (int i = 0; i < m_dev_mgr->get_local_count(); ++i) {
        Device& dev = m_dev_mgr->get_device(i);
        auto& part = m_buf_mgr->get_partition(i);
        
        cl_event exec_event;
        
        // Disparo com g_offset e g_size corretos para cada dispositivo
        cl_int err = clEnqueueNDRangeKernel(dev.kernelQueue, 
                                            dev.kernels[m_current_kernel_id], 1, 
                                            &part.offset, &part.length, nullptr, 
                                            0, nullptr, &exec_event);

        if (err == CL_SUCCESS) {
            // ALOCAÇÃO DINÂMICA ÚNICA: Fundamental para não sobrescrever dados no callback
            ProfilingData* p_data = new ProfilingData();
            p_data->device_id = i;
            p_data->kernel_id = m_current_kernel_id;
            p_data->global_size = part.length;
            p_data->event = exec_event; 
            p_data->rank = m_rank;

            // Registrar o callback
            clSetEventCallback(exec_event, CL_COMPLETE, execution_complete_callback, p_data);
            
            // Forçar o envio para o hardware
            clFlush(dev.kernelQueue);
        }
    }
}


void Runtime::gather_global(int global_id, void* target_host_ptr) {
    // O BufferManager deve realizar o clEnqueueReadBuffer e o MPI_Gather
    m_buf_mgr->gather_global(global_id, target_host_ptr, *m_dev_mgr);
}


void Runtime::wait_all() {
    for (int i = 0; i < m_dev_mgr->get_local_count(); ++i) {
        // clFinish garante que o dispositivo i terminou tudo antes da CPU prosseguir
        clFinish(m_dev_mgr->get_device(i).kernelQueue);
    }
}

int Runtime::get_local_device_count() const {
    return m_dev_mgr->get_local_count();
}

}