#include "core/device_manager.hpp"
#include <iostream>
#include <vector>

namespace dcl {

void DeviceManager::initialize(DeviceTag tag, int max_devices) {
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "[DCL ERROR] Nenhuma plataforma OpenCL encontrada." << std::endl;
        return;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    // Mapeamento da Tag para o tipo de dispositivo OpenCL
    cl_device_type type = (tag == DeviceTag::GPU) ? CL_DEVICE_TYPE_GPU : 
                          (tag == DeviceTag::CPU) ? CL_DEVICE_TYPE_CPU : 
                                                    CL_DEVICE_TYPE_ALL;

    for (auto p : platforms) {
        std::vector<cl_device_id> platform_devices;
        cl_uint dev_count;
        cl_device_id ids[10]; 
        
        // Busca dispositivos do tipo solicitado nesta plataforma
        err = clGetDeviceIDs(p, type, 10, ids, &dev_count);
        if (err != CL_SUCCESS || dev_count == 0) continue;

        // Limita ao número máximo solicitado pelo usuário
        for (cl_uint i = 0; i < dev_count && platform_devices.size() < (size_t)max_devices; ++i) {
            platform_devices.push_back(ids[i]);
        }

        if (platform_devices.empty()) continue;

        // 1. PROPRIEDADES DO CONTEXTO (Usa cl_context_properties)
        cl_context_properties ctx_props[] = { 
            CL_CONTEXT_PLATFORM, (cl_context_properties)p, 
            0 
        };

        cl_context shared_ctx = clCreateContext(ctx_props, (cl_uint)platform_devices.size(), 
                                               platform_devices.data(), nullptr, nullptr, &err);
        
        if (err != CL_SUCCESS) {
            std::cerr << "[DCL ERROR] Falha ao criar contexto compartilhado: " << err << std::endl;
            continue;
        }

        // 2. PROPRIEDADES DA FILA (Usa cl_queue_properties)
        // O erro C167 ocorria aqui por usar ctx_props em vez de q_props
        const cl_queue_properties q_props[] = { 
            CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 
            0 
        };

        for (auto d_id : platform_devices) {
            Device dev;
            dev.deviceID = d_id;
            dev.context = shared_ctx; // Todos compartilham o mesmo contexto no nó

            // Criar filas individuais para Kernel e Data com Profiling Habilitado
            dev.kernelQueue = clCreateCommandQueueWithProperties(shared_ctx, d_id, q_props, &err);
            dev.dataQueue = clCreateCommandQueueWithProperties(shared_ctx, d_id, q_props, &err);
            
            if (err != CL_SUCCESS) {
                std::cerr << "[DCL ERROR] Falha ao criar filas para o dispositivo: " << err << std::endl;
                continue;
            }

            // Coleta informações de hardware para o balanceamento de carga (TCC)
            clGetDeviceInfo(d_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(dev.computeUnits), &dev.computeUnits, nullptr);
            
            m_local_devices.push_back(dev);
        }

        // Uma vez que configuramos uma plataforma com sucesso, interrompemos a busca
        if (!m_local_devices.empty()) break;
    }

    if (m_local_devices.empty()) {
        std::cerr << "[DCL WARNING] Nenhum dispositivo inicializado para a Tag selecionada." << std::endl;
    } else {
        std::cout << "[DCL INFO] " << m_local_devices.size() << " dispositivo(s) inicializado(s) no Rank." << std::endl;
    }
}

} // namespace dcl