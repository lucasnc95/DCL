#include<CL/cl.h>
#include <vector>
#include <unordered_map>


namespace dcl {
    struct Device {
        cl_device_id deviceID;
        cl_device_type deviceType;
        cl_context context;
        cl_command_queue kernelQueue;
        cl_command_queue dataQueue;
        cl_program program;
        
        // Indexação local para controle total (Ref: struct Device original)
        std::unordered_map<int, cl_mem> memoryObjects; // GlobalID -> cl_mem
        std::unordered_map<int, cl_kernel> kernels;    // KernelID -> cl_kernel
        std::vector<cl_event> profilingEvents;

        cl_uint computeUnits;
        size_t maxWorkGroupSize;

        // Métodos de Profiling (Baseados no GetEventTaskTicks original)
        long get_task_ticks(int event_idx);
        long get_overhead_ticks(int event_idx);
    };
}