#ifndef DCL_KERNEL_MANAGER_HPP
#define DCL_KERNEL_MANAGER_HPP

#include "device_manager.hpp"
#include <string> 

namespace dcl {
    class KernelManager {
    public:
        int build_program(DeviceManager& dev_mgr, const std::string& path, const std::string& name);
    private:
        int m_kernel_counter = 0;
    };
}

#endif