#include "core/device_manager.hpp"
#include <iostream>
#include <vector>

namespace dcl {

void DeviceManager::initialize(DeviceTag tag, int max_devices, bool verbose) {
    cl_int state;
    cl_uint numPlatforms = 0;
    
    // 1) Obter todas as plataformas disponíveis
    // Usamos um valor alto (ex: 10) para garantir que pegamos Intel, NVIDIA, AMD, etc.
    cl_platform_id platformIDs[10];
    state = clGetPlatformIDs(10, platformIDs, &numPlatforms);
    if (state != CL_SUCCESS || numPlatforms == 0) {
        if (verbose) std::cerr << "OpenCL Error: Platforms couldn't be found." << std::endl;
        return;
    }

    // 2) Definir o tipo de dispositivo (Inspirado no seu device_types.compare)
    cl_device_type selType = CL_DEVICE_TYPE_ALL;
    if (tag == DeviceTag::GPU) selType = CL_DEVICE_TYPE_GPU;
    else if (tag == DeviceTag::CPU) selType = CL_DEVICE_TYPE_CPU;

    // 3) Loop por TODAS as plataformas (Removido o break anterior)
    for (cl_uint p = 0; p < numPlatforms; ++p) {
        cl_uint cnt = 0;
        cl_device_id tmp[10]; // Máximo de dispositivos por plataforma
        
        state = clGetDeviceIDs(platformIDs[p], selType, 10, tmp, &cnt);
        if (state != CL_SUCCESS || cnt == 0) continue;

        // Criar um contexto para todos os dispositivos desta plataforma específica
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[p],
            0
        };
        cl_context ctx = clCreateContext(props, cnt, tmp, NULL, NULL, &state);
        
        if (state != CL_SUCCESS) continue;

        // 4) Criar a fila de comando (Command Queue) com Profiling
        // Seguindo sua lib: uma fila por contexto (que atende ao primeiro device tmp[0])
        cl_queue_properties q_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, tmp[0], q_props, &state);
        
        if (state != CL_SUCCESS) {
            // Fallback OpenCL 1.2 como na sua lib antiga
            queue = clCreateCommandQueue(ctx, tmp[0], CL_QUEUE_PROFILING_ENABLE, &state);
        }

        // 5) Adicionar cada dispositivo ao nosso gerenciador local
        for (cl_uint j = 0; j < cnt && m_local_devices.size() < (size_t)max_devices; ++j) {
            Device dev;
            dev.deviceID = tmp[j];
            dev.context = ctx;
            dev.kernelQueue = queue; // Mesma fila para kernel e data como na lib antiga
            dev.dataQueue = queue;
            dev.program = NULL;

            // Obter metadados para o verbose e balanceamento
            char name[128];
            clGetDeviceInfo(tmp[j], CL_DEVICE_NAME, sizeof(name), name, NULL);
            clGetDeviceInfo(tmp[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(dev.computeUnits), &dev.computeUnits, NULL);

            if (verbose) {
                printf("[DCL INFO] Rank %d - Device (%zu): %s, CUs=%u\n", 
                       m_rank, m_local_devices.size(), name, dev.computeUnits);
            }

            m_local_devices.push_back(dev);
        }
    }

    // 6) Lógica de Sincronização MPI (Igual ao seu InitDevices)
    int local_count = (int)m_local_devices.size();
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::vector<int> devicesWorld(world_size, 0);
    // Usamos Allgather para que todos saibam quantos dispositivos cada rank tem
    MPI_Allgather(&local_count, 1, MPI_INT, devicesWorld.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int todosDispositivos = 0;
    for (int count = 0; count < world_size; count++) {
        if (count == m_rank) {
            m_offset = todosDispositivos; // meusDispositivosOffset
            m_length = devicesWorld[count]; // meusDispositivosLength
        }
        todosDispositivos += devicesWorld[count];
    }

    if (verbose && m_rank == 0) {
        printf("[DCL INFO] Total de dispositivos no Cluster: %d\n", todosDispositivos);
    }
}

} // namespace dcl