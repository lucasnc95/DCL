#ifndef DCL_DEVICE_MANAGER_HPP
#define DCL_DEVICE_MANAGER_HPP

#include <CL/cl.h>
#include <vector>
#include <unordered_map>
#include "types.hpp"

namespace dcl {

struct Device {
    cl_device_id deviceID;
    cl_device_type deviceType;
    cl_context context;
    cl_command_queue kernelQueue;
    cl_command_queue dataQueue;
    cl_program program;
    
    // Mapeamento GlobalID -> cl_mem (Substitui o memoryObjectID original)
    std::unordered_map<int, cl_mem> memoryObjects;
    std::unordered_map<int, cl_kernel> kernels;
    std::vector<cl_event> events; // Pool de eventos para profiling

    cl_uint computeUnits;
    size_t maxWorkGroupSize;
    cl_event last_execution_event = nullptr;
};

class DeviceManager {
public:
    // Inicializa dispositivos baseando-se na tag (CPU, GPU ou ALL)
    void initialize(DeviceTag tag, int max_devices);
    
    Device& get_device(int local_idx) { return m_local_devices[local_idx]; }
    int get_local_count() const { return static_cast<int>(m_local_devices.size()); }
    cl_context get_shared_context() { return m_local_devices.empty() ? nullptr : m_local_devices[0].context; }

private:
    std::vector<Device> m_local_devices; // Membro agora definido corretamente
};

} // namespace dcl
#endif