#include <mpi.h>
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

struct DadoBruto {
    std::size_t volume;
    double tempo_migracao;
};

struct OclContext {
    bool valid = false;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_device_id device = nullptr;
};

// ============================================================================
// 1. INICIALIZAÇÃO DO OPENCL
// ============================================================================
OclContext Inicializar_OpenCL(int meu_rank) {
    OclContext ocl;
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms == 0) return ocl;

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    for (auto plat : platforms) {
        cl_uint num_devices = 0;
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        if (num_devices == 0) continue;

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
        
        ocl.device = devices[0]; // Pega o primeiro dispositivo disponível
        
        cl_int err;
        ocl.context = clCreateContext(nullptr, 1, &ocl.device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) continue;

#if CL_TARGET_OPENCL_VERSION >= 200
        ocl.queue = clCreateCommandQueueWithProperties(ocl.context, ocl.device, nullptr, &err);
#else
        ocl.queue = clCreateCommandQueue(ocl.context, ocl.device, 0, &err);
#endif
        if (err == CL_SUCCESS) {
            ocl.valid = true;
            if (meu_rank == 0) std::cout << "OpenCL detectado. Profiling incluira latencia de barramento PCIe." << std::endl;
            break;
        }
    }
    return ocl;
}

// ============================================================================
// 2. GERAÇÃO DO VETOR
// ============================================================================
std::vector<std::size_t> Gerar_Vetor_Volumes() {
    std::vector<std::size_t> vetor = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 
                                      1500, 2048, 4096, 8192, 16384, 32768, 65536, 
                                      131072, 262144, 524288, 1048576, 4194304, 16777216};

    return vetor;
}

// ============================================================================
// 3. PING-PONG
// ============================================================================
DadoBruto Run_Ping_Pong(std::size_t volume, int meu_rank, const OclContext& ocl) {
    std::vector<char> host_send(volume, 'A');
    std::vector<char> host_recv(volume, 'B');
    
    cl_mem dev_send = nullptr;
    cl_mem dev_recv = nullptr;
    cl_int err;

    // Se houver GPU, aloca na VRAM
    if (ocl.valid && volume > 0) {
        dev_send = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, volume, nullptr, &err);
        dev_recv = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, volume, nullptr, &err);
    }

    int iteracoes = (volume < 1048576) ? 500 : 50; 
    MPI_Barrier(MPI_COMM_WORLD);

    double tempo_inicio = MPI_Wtime();
    
    for(int i = 0; i < iteracoes; i++) {
        if (meu_rank == 0) {
            // 1. Lê da GPU para o Host
            if (ocl.valid && volume > 0) {
                clEnqueueReadBuffer(ocl.queue, dev_send, CL_TRUE, 0, volume, host_send.data(), 0, nullptr, nullptr);
            }
            // 2. Envia e Recebe via Rede
            MPI_Send(host_send.data(), volume, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(host_recv.data(), volume, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // 3. Escreve do Host para a GPU
            if (ocl.valid && volume > 0) {
                clEnqueueWriteBuffer(ocl.queue, dev_recv, CL_TRUE, 0, volume, host_recv.data(), 0, nullptr, nullptr);
            }
        } else if (meu_rank == 1) {
            MPI_Recv(host_recv.data(), volume, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (ocl.valid && volume > 0) {
                clEnqueueWriteBuffer(ocl.queue, dev_recv, CL_TRUE, 0, volume, host_recv.data(), 0, nullptr, nullptr);
                clEnqueueReadBuffer(ocl.queue, dev_send, CL_TRUE, 0, volume, host_send.data(), 0, nullptr, nullptr);
            }
            MPI_Send(host_send.data(), volume, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    double tempo_fim = MPI_Wtime();

    if (ocl.valid && volume > 0) {
        clReleaseMemObject(dev_send);
        clReleaseMemObject(dev_recv);
    }

    DadoBruto resultado = {volume, 0.0};

    if (meu_rank == 0) {
        double rtt_medio = (tempo_fim - tempo_inicio) / iteracoes;
        resultado.tempo_migracao = rtt_medio / 2.0; // Pega exatamente o tempo de 1 via
        std::cout << "Coletado volume: " << volume << " bytes." << std::endl;
    }
    return resultado;
}

// ============================================================================
// 4. FLUXO PRINCIPAL
// ============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int meu_rank, total_processos;
    MPI_Comm_rank(MPI_COMM_WORLD, &meu_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processos);

    if (total_processos < 2) {
        if (meu_rank == 0) std::cerr << "Erro: Requer no mínimo 2 processos (mpirun -np 2)." << std::endl;
        MPI_Finalize();
        return 1;
    }

    OclContext ocl = Inicializar_OpenCL(meu_rank);
    
    std::vector<std::size_t> vetor_de_testes;
    int tamanho_do_vetor = 0;

    if (meu_rank == 0) {
        vetor_de_testes = Gerar_Vetor_Volumes();
        tamanho_do_vetor = static_cast<int>(vetor_de_testes.size());
    }

    MPI_Bcast(&tamanho_do_vetor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (meu_rank != 0) vetor_de_testes.resize(tamanho_do_vetor);

    MPI_Bcast(vetor_de_testes.data(), tamanho_do_vetor * sizeof(std::size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    std::vector<DadoBruto> dados_brutos;
    dados_brutos.reserve(static_cast<std::size_t>(tamanho_do_vetor));

    for (std::size_t volume_atual : vetor_de_testes) {
        dados_brutos.push_back(Run_Ping_Pong(volume_atual, meu_rank, ocl));
    }

    if (meu_rank == 0) {
        std::ofstream arquivo_saida("profiling_results.txt");
        arquivo_saida << "Max_Volume_Bytes\tm_TempoPorByte\tb_Latencia\n";

        for (std::size_t i = 0; i < dados_brutos.size() - 1; ++i) {
            const auto& p1 = dados_brutos[i];
            const auto& p2 = dados_brutos[i + 1];

            double m = 0.0;
            double delta_v = static_cast<double>(p2.volume - p1.volume);
            
            if (delta_v > 0.0) {
                m = (p2.tempo_migracao - p1.tempo_migracao) / delta_v;
                if (m < 0.0) m = 0.0;
            }

            double b = p1.tempo_migracao - m * static_cast<double>(p1.volume);
            if (b < 0.0) b = 0.0; // Garante que não haja latência negativa por ruído de CPU

            arquivo_saida << p2.volume << "\t" << m << "\t" << b << "\n";
        }
        arquivo_saida.close();
        std::cout << "\nProfiling concluído! Segmentos lineares salvos em 'profiling_results.txt'." << std::endl;
    }

    if (ocl.valid) {
        clReleaseCommandQueue(ocl.queue);
        clReleaseContext(ocl.context);
    }

    MPI_Finalize();
    return 0;
}