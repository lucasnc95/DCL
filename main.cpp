// #include "include/dcl/runtime.hpp"
// #include <mpi.h>

// #include <cstdio>
// #include <cstdlib>
// #include <cstring>
// #include <iostream>
// #include <optional>
// #include <vector>
// #include <chrono>

// // Tipos de celulas.
// #define CELULA_A            0
// #define CELULA_MR           1
// #define CELULA_MA           2
// #define CELULA_N            3
// #define CELULA_CH           4
// #define CELULA_ND           5
// #define CELULA_G            6
// #define CELULA_CA           7
// #define MALHA_TOTAL_CELULAS 8

// // Informacoes de acesso à estrutura "parametrosMalha".
// #define OFFSET_COMPUTACAO               0
// #define LENGTH_COMPUTACAO               1
// #define COMPRIMENTO_GLOBAL_X            2
// #define COMPRIMENTO_GLOBAL_Y            3
// #define COMPRIMENTO_GLOBAL_Z            4
// #define MALHA_DIMENSAO_POSICAO_Z        5
// #define MALHA_DIMENSAO_POSICAO_Y        6
// #define MALHA_DIMENSAO_POSICAO_X        7
// #define MALHA_DIMENSAO_CELULAS          8
// #define NUMERO_PARAMETROS_MALHA         9

// static inline std::size_t idx_his(
//     int celula,
//     int x,
//     int y,
//     int z,
//     const int* parametrosMalha
// ) {
//     return static_cast<std::size_t>(celula * parametrosMalha[MALHA_DIMENSAO_CELULAS]) +
//            static_cast<std::size_t>(z) * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z] +
//            static_cast<std::size_t>(y) * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y] +
//            static_cast<std::size_t>(x) * parametrosMalha[MALHA_DIMENSAO_POSICAO_X];
// }

// void InicializarParametrosMalhaHIS(
//     int* parametrosMalha,
//     int offsetComputacao,
//     int lengthComputacao,
//     int xMalhaLength,
//     int yMalhaLength,
//     int zMalhaLength
// ) {
//     parametrosMalha[OFFSET_COMPUTACAO] = offsetComputacao;
//     parametrosMalha[LENGTH_COMPUTACAO] = lengthComputacao;
//     parametrosMalha[COMPRIMENTO_GLOBAL_X] = xMalhaLength;
//     parametrosMalha[COMPRIMENTO_GLOBAL_Y] = yMalhaLength;
//     parametrosMalha[COMPRIMENTO_GLOBAL_Z] = zMalhaLength;
//     parametrosMalha[MALHA_DIMENSAO_POSICAO_Z] = yMalhaLength * xMalhaLength * MALHA_TOTAL_CELULAS;
//     parametrosMalha[MALHA_DIMENSAO_POSICAO_Y] = xMalhaLength * MALHA_TOTAL_CELULAS;
//     parametrosMalha[MALHA_DIMENSAO_POSICAO_X] = MALHA_TOTAL_CELULAS;
//     parametrosMalha[MALHA_DIMENSAO_CELULAS] = 1;
// }

// void InicializarPontosHIS(float* malha, int* parametrosMalha) {
//     for (int x = 0; x < parametrosMalha[COMPRIMENTO_GLOBAL_X]; ++x) {
//         for (int y = 0; y < parametrosMalha[COMPRIMENTO_GLOBAL_Y]; ++y) {
//             for (int z = 0; z < parametrosMalha[COMPRIMENTO_GLOBAL_Z]; ++z) {
//                 if (z >= static_cast<int>(0.75f * parametrosMalha[COMPRIMENTO_GLOBAL_Z])) {
//                     malha[idx_his(CELULA_A, x, y, z, parametrosMalha)] = 100.0f;
//                 } else {
//                     malha[idx_his(CELULA_A, x, y, z, parametrosMalha)] = 0.0f;
//                 }

//                 malha[idx_his(CELULA_MR, x, y, z, parametrosMalha)] = 1.0f;
//                 malha[idx_his(CELULA_MA, x, y, z, parametrosMalha)] = 0.0f;
//                 malha[idx_his(CELULA_N,  x, y, z, parametrosMalha)] = 0.0f;
//                 malha[idx_his(CELULA_CH, x, y, z, parametrosMalha)] = 0.0f;
//                 malha[idx_his(CELULA_ND, x, y, z, parametrosMalha)] = 0.0f;
//                 malha[idx_his(CELULA_G,  x, y, z, parametrosMalha)] = 0.0f;
//                 malha[idx_his(CELULA_CA, x, y, z, parametrosMalha)] = 0.0f;
//             }
//         }
//     }
// }

// void PrintMalhaCompletaUnida(float* malha, int* parametrosMalha, const char* nome)
// {
//     int X = parametrosMalha[COMPRIMENTO_GLOBAL_X];
//     int Y = parametrosMalha[COMPRIMENTO_GLOBAL_Y];
//     int Z = parametrosMalha[COMPRIMENTO_GLOBAL_Z];

//     std::cout << "\n=== " << nome << " ===\n";

//     for (int x = 0; x < X; x++) {
//         for (int y = 0; y < Y; y++) {
//             for (int z = 0; z < Z; z++) {
//                 float v = malha[
//                     (CELULA_A * parametrosMalha[MALHA_DIMENSAO_CELULAS]) +
//                     (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) +
//                     (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) +
//                     (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X])
//                 ];
//                 std::printf("%8.4f ", v);
//             }
//             std::printf("\n");
//         }
//         std::printf("\n");
//     }
// }

// static void print_partitions(const std::vector<dcl::DevicePartition>& parts) {
//     std::cout << "=== Particoes globais ===\n";
//     for (std::size_t i = 0; i < parts.size(); ++i) {
//         const dcl::DevicePartition& p = parts[i];
//         std::cout
//             << "part[" << i << "] "
//             << "device_global=" << p.device_global_index
//             << " rank=" << p.owning_rank
//             << " local_index=" << p.local_index
//             << " offset=" << p.global_offset
//             << " count=" << p.element_count
//             << "\n";
//     }
//     std::cout << "\n";
// }

// int main(int argc, char** argv) {
//     try {
//         std::chrono::steady_clock sc;   
//         auto start = sc.now();  
//         auto runtime = dcl::Runtime::create(argc, argv);

//         runtime.discover_devices({
//             dcl::DeviceKind::cpu,
//             0
//         });

//         if (runtime.rank() == 0) {
//             std::cout << "=== Dispositivos locais do rank 0 ===\n";
//             const std::vector<dcl::DeviceInfo>& devs = runtime.devices();
//             for (std::size_t i = 0; i < devs.size(); ++i) {
//                 const dcl::DeviceInfo& d = devs[i];
//                 std::cout
//                     << "global=" << d.global_index
//                     << " local=" << d.local_index
//                     << " nome=\"" << d.name << "\""
//                     << " compute_units=" << d.compute_units
//                     << "\n";
//             }
//             std::cout << "\n";
//         }

//         const int x = 10;
//         const int y = 10;
//         const int z = 10;
//         const int iterations = 10000;

//         const int total_elements = x * y * z;
//         const std::size_t tam = static_cast<std::size_t>(total_elements) * MALHA_TOTAL_CELULAS;

//         std::vector<int> parametros(NUMERO_PARAMETROS_MALHA, 0);
//         std::vector<float> malha(tam, 0.0f);

//         InicializarParametrosMalhaHIS(parametros.data(), 0, total_elements, x, y, z);
//         InicializarPontosHIS(malha.data(), parametros.data());

//         // 1) KERNEL PRIMEIRO
//         auto kernel = runtime.create_kernel({
//             "kernels.cl",
//             "ProcessarPontos",
//             ""
//         });

//         // 2) PARTICIONAMENTO DEPOIS
//         const std::size_t sub = static_cast<std::size_t>(x) * static_cast<std::size_t>(y);

//         runtime.set_partition({
//             static_cast<std::size_t>(total_elements),
//             MALHA_TOTAL_CELULAS,
//             sizeof(float),
//             sub
//         });

//         if (runtime.rank() == 0) {
//             print_partitions(runtime.partitions());
//         }

//         // 3) BUFFERS/FIELDS SÓ AGORA
//         auto params_buf = runtime.create_buffer({
//             "parametros",
//             static_cast<std::size_t>(NUMERO_PARAMETROS_MALHA) * sizeof(int),
//             dcl::BufferUsage::read_only,
//             parametros.data()
//         });

//         auto state_a = runtime.create_field({
//             "malha_a",
//             static_cast<std::size_t>(total_elements),
//             MALHA_TOTAL_CELULAS,
//             sizeof(float),
//             dcl::BufferUsage::read_write,
//             malha.data()
//         });

//         auto state_b = runtime.create_field({
//             "malha_b",
//             static_cast<std::size_t>(total_elements),
//             MALHA_TOTAL_CELULAS,
//             sizeof(float),
//             dcl::BufferUsage::read_write,
//             malha.data()
//         });

//         auto bind_a_from_b = runtime.bind(kernel)
//             .arg(0, state_a)
//             .arg(1, state_b)
//             .arg(2, params_buf)
//             .build();

//         auto bind_b_from_a = runtime.bind(kernel)
//             .arg(0, state_b)
//             .arg(1, state_a)
//             .arg(2, params_buf)
//             .build();

//         for (int iter = 0; iter < iterations; ++iter) {
//             const bool even = (iter % 2 == 0);

//             const dcl::KernelBinding& binding = even ? bind_a_from_b : bind_b_from_a;
//             const dcl::FieldHandle input_field = even ? state_b : state_a;

//             dcl::ExecutionStep step = runtime.step("his-step")
//                 .invoke(
//                     binding,
//                     dcl::LaunchGeometry{
//                         0,
//                         static_cast<std::size_t>(total_elements),
//                         std::optional<std::size_t>()
//                     }
//                 )
//                 .with_halo_exchange(dcl::HaloSpec{
//                     sub,
//                     std::vector<dcl::FieldHandle>{input_field}
//                 })
//                 .with_balance(dcl::AutoBalancePolicy{
//                     dcl::BalanceMode::off,
//                     0
//                 })
//                 .synchronize_at_end(true)
//                 .build();

//             runtime.execute(step);

//             std::vector<float> gathered_a(tam, 0.0f);
//             std::vector<float> gathered_b(tam, 0.0f);

//             runtime.gather(state_a, gathered_a.data(), gathered_a.size() * sizeof(float));
//             runtime.gather(state_b, gathered_b.data(), gathered_b.size() * sizeof(float));

//             if (runtime.rank() == 0 && iter == iterations -1) {
//                 std::cout << "\n============================\n";
//                 std::cout << "ITERACAO " << (iter + 1) << "\n";                
//                 PrintMalhaCompletaUnida(gathered_a.data(), parametros.data(), "STATE_A");
//                 PrintMalhaCompletaUnida(gathered_b.data(), parametros.data(), "STATE_B");
//                 auto end = sc.now();
//                 std::chrono::duration<double> elapsed_seconds = end - start;
//                 std::cout << "Tempo de execução: " << elapsed_seconds << "\n";
//             }
//         }

//         return 0;
//     } catch (const dcl::Error& e) {
//         std::cerr << "dcl error: " << e.what() << std::endl;
//         return 1;
//     } catch (const std::exception& e) {
//         std::cerr << "std error: " << e.what() << std::endl;
//         return 2;
//     }
// }

#include "include/dcl/runtime.hpp"
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

int main(int argc, char** argv) {
    try {
        auto runtime = dcl::Runtime::create(argc, argv);

        runtime.discover_devices({
            dcl::DeviceKind::all,
            0
        });

        const int nx = 512;
        const int ny = 512;
        const int nz = 512;
        const int iterations = 200;
        const int flop_repeat = 64;

        const std::size_t n = static_cast<std::size_t>(nx) * ny * nz;
        std::vector<float> a(n, 0.0f);
        std::vector<float> b(n, 0.0f);

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const std::size_t idx =
                        static_cast<std::size_t>(z) * nx * ny +
                        static_cast<std::size_t>(y) * nx + x;
                    a[idx] = 0.001f * float((x + y + z) % 251);
                }
            }
        }

        auto kernel = runtime.create_kernel({
            "jacobi3d.cl",
            "jacobi3d_7p",
            ""
        });

        // halo de 1 plano em Z
        const std::size_t halo = static_cast<std::size_t>(nx) * ny;

        runtime.set_partition({
            n,
            1,
            sizeof(float),
            halo
        });

        auto field_a = runtime.create_field({
            "field_a",
            n,
            1,
            sizeof(float),
            dcl::BufferUsage::read_write,
            a.data()
        });

        auto field_b = runtime.create_field({
            "field_b",
            n,
            1,
            sizeof(float),
            dcl::BufferUsage::read_write,
            b.data()
        });

        auto bind_b_from_a = runtime.bind(kernel)
            .arg(0, field_b)
            .arg(1, field_a)
            .arg(2, dcl::ScalarArg(nx))
            .arg(3, dcl::ScalarArg(ny))
            .arg(4, dcl::ScalarArg(nz))
            .arg(5, dcl::ScalarArg(nx))
            .arg(6, dcl::ScalarArg(nx * ny))
            .arg(7, dcl::ScalarArg(flop_repeat))
            .build();

        auto bind_a_from_b = runtime.bind(kernel)
            .arg(0, field_a)
            .arg(1, field_b)
            .arg(2, dcl::ScalarArg(nx))
            .arg(3, dcl::ScalarArg(ny))
            .arg(4, dcl::ScalarArg(nz))
            .arg(5, dcl::ScalarArg(nx))
            .arg(6, dcl::ScalarArg(nx * ny))
            .arg(7, dcl::ScalarArg(flop_repeat))
            .build();

        dcl::ExecutionStep step_b_from_a = runtime.step("jacobi_b_from_a")
            .invoke(
                bind_b_from_a,
                dcl::LaunchGeometry{
                    0,
                    n,
                    std::nullopt
                }
            )
            .with_halo_exchange(dcl::HaloSpec{
                halo,
                std::vector<dcl::FieldHandle>{field_a}
            })
            .with_balance(dcl::AutoBalancePolicy{
                dcl::BalanceMode::off,
                0
            })
            .synchronize_at_end(true)
            .build();

        dcl::ExecutionStep step_a_from_b = runtime.step("jacobi_a_from_b")
            .invoke(
                bind_a_from_b,
                dcl::LaunchGeometry{
                    0,
                    n,
                    std::nullopt
                }
            )
            .with_halo_exchange(dcl::HaloSpec{
                halo,
                std::vector<dcl::FieldHandle>{field_b}
            })
            .with_balance(dcl::AutoBalancePolicy{
                dcl::BalanceMode::off,
                0
            })
            .synchronize_at_end(true)
            .build();

        MPI_Barrier(runtime.communicator());
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int it = 0; it < iterations; ++it) {
            if ((it % 2) == 0) {
                runtime.execute(step_b_from_a);
            } else {
                runtime.execute(step_a_from_b);
            }
        }

        MPI_Barrier(runtime.communicator());
        auto t1 = std::chrono::high_resolution_clock::now();

        std::vector<float> gathered(n, 0.0f);
        if ((iterations % 2) == 0) {
            runtime.gather(field_a, gathered.data(), gathered.size() * sizeof(float));
        } else {
            runtime.gather(field_b, gathered.data(), gathered.size() * sizeof(float));
        }

        double checksum = 0.0;
        if (runtime.rank() == 0) {
            for (std::size_t i = 0; i < gathered.size(); ++i) checksum += gathered[i];
        }

        const double seconds = std::chrono::duration<double>(t1 - t0).count();

        if (runtime.rank() == 0) {
            std::cout << "DCL\n";
            std::cout << "tempo_total_s=" << seconds << "\n";
            std::cout << "tempo_medio_iter_s=" << (seconds / iterations) << "\n";
            std::cout << "checksum=" << checksum << "\n";
        }

        return 0;
    } catch (const dcl::Error& e) {
        std::cerr << "dcl error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "std error: " << e.what() << "\n";
        return 2;
    }
}

