#ifndef DCL_RUNTIME_HPP
#define DCL_RUNTIME_HPP

#include "../core/device_manager.hpp"
#include "../core/kernel_manager.hpp"
#include "../data/buffer_manager.hpp"
#include "../comm/halo_manager.hpp"
#include "../data/dependency.hpp"
#include <memory>
#include <string>

namespace dcl {
    class Runtime {
    public:
        Runtime(int &argc, char** &argv);
        ~Runtime();

        void init_devices(DeviceTag tag, int max_devs, bool verbose = false);
        void create_kernel(const std::string& file, const std::string& name);
        
        int create_buffer(size_t total_elements, cl_mem_flags flags);
        void write_buffer(int global_id, const void* data, size_t offset, size_t size);
        void gather_global(int global_id, void* target_host_ptr);
        void wait_all();
        
        // Retorna quantos dispositivos este rank controla
        int get_local_device_count() const;
        
        // As duas versões de set_arg exigidas pelo Linker
        void set_arg(int arg_idx, int global_buffer_id);
        void set_arg(int arg_idx, void* scalar_data, size_t size);
        
        void enqueue_kernel(const DataDependency& dep);

        int rank() const { return m_rank; }
        int size() const { return m_size; }

    private:
        int m_rank, m_size;
        int m_current_kernel_id = -1;
        std::unique_ptr<DeviceManager> m_dev_mgr;
        std::unique_ptr<KernelManager> m_kernel_mgr;
        std::unique_ptr<BufferManager> m_buf_mgr;
        std::unique_ptr<HaloManager> m_halo_mgr; 
    };


}
#endif