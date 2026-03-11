#include "core/kernel_manager.hpp"
#include <fstream>
#include <iostream>

namespace dcl {

int KernelManager::build_program(DeviceManager& dev_mgr, const std::string& path, const std::string& name) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERRO: Nao foi possivel abrir o arquivo do kernel: " << path << std::endl;
        return -1;
    }
    std::string src((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char* src_ptr = src.c_str();
    size_t src_len = src.length();

    cl_int err;
    cl_context ctx = dev_mgr.get_shared_context();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);

    // Tenta buildar para todos os dispositivos do contexto
    err = clBuildProgram(prog, 0, nullptr, nullptr, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        // ESSENCIAL: Capturar o log de erro do driver para evitar o SegFault silencioso
        char build_log[16384];
        clGetProgramBuildInfo(prog, dev_mgr.get_device(0).deviceID, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, nullptr);
        std::cerr << "--- LOG DE ERRO DO KERNEL ---" << std::endl << build_log << std::endl;
        return -1;
    }

    int kernel_id = m_kernel_counter++;
    for (int i = 0; i < dev_mgr.get_local_count(); ++i) {
        Device& dev = dev_mgr.get_device(i);
        dev.program = prog; // Mantem referencia
        dev.kernels[kernel_id] = clCreateKernel(prog, name.c_str(), &err);
    }
    return kernel_id;
}
}