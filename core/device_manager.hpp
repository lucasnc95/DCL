#ifndef DCL_DEVICE_MANAGER_HPP
#define DCL_DEVICE_MANAGER_HPP

#include <CL/cl.h>
#include <vector>
#include <unordered_map>
#include "types.hpp"

namespace dcl {
    // Tags para seleção de hardware
    enum class DeviceTag { CPU, GPU, ALL };

    struct Device {
        cl_device_id deviceID;
        cl_context context;
        cl_command_queue kernelQueue;
        cl_command_queue dataQueue;
        cl_program program;
        std::map<int, cl_kernel> kernels;
        std::map<int, cl_mem> memoryObjects;
        cl_uint computeUnits;
        cl_device_type deviceType;
    };

    class DeviceManager {
    public:
        DeviceManager() : m_rank(0), m_offset(0), m_length(0) {}
        
        void initialize(DeviceTag tag, int max_devices, bool verbose = false);
        
        int get_local_count() const { return (int)m_local_devices.size(); }
        Device& get_device(int index) { return m_local_devices[index]; }
        
        // Getters para a lógica de partição global (TCC UFJF)
        int get_global_offset() const { return m_offset; }
        int get_global_length() const { return m_length; }

    private:
        std::vector<Device> m_local_devices;
        int m_rank;     // Rank do processo MPI
        int m_offset;   // Offset global de dispositivos (meusDispositivosOffset)
        int m_length;   // Quantidade de dispositivos locais (meusDispositivosLength)
    };
}
#endif