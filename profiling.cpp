#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cstddef>

// ============================================================================
// 1. CRIAÇÃO DO VETOR VOLUMES (Focado apenas em TCP/IP)
// ============================================================================
std::vector<std::size_t> Gerar_Vetor_Volumes() {
    // Potências de 2 fundamentais para medir latência pura e transição de protocolo
    std::vector<std::size_t> vetor_base = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    // Potências de 2 para medir banda - de 16KB a 16MB
    std::vector<std::size_t> vetor_grandes = {
        16384, 32768, 65536, 131072, 262144, 524288, 1048576, 4194304, 16777216
    };

    std::vector<std::size_t> tcp_mtu = {1500, 2048, 4096, 8192};

    vetor_base.insert(vetor_base.end(), tcp_mtu.begin(), tcp_mtu.end());
    vetor_base.insert(vetor_base.end(), vetor_grandes.begin(), vetor_grandes.end());

    std::sort(vetor_base.begin(), vetor_base.end());
    vetor_base.erase(std::unique(vetor_base.begin(), vetor_base.end()), vetor_base.end());

    return vetor_base;
}

struct DadoBruto {
    std::size_t volume;
    double tempo_rede;
    double tempo_mem;
};

// ============================================================================
// 2. EXECUÇÃO DO PING-PONG E MEMÓRIA
// ============================================================================
DadoBruto Executar_Ping_Pong_E_Medir_Tempo(std::size_t volume, int meu_rank) {
    DadoBruto resultado{volume, 0.0, 0.0};

    // Buffers para o teste de rede
    std::vector<char> send_buf(volume, 'A');
    std::vector<char> recv_buf(volume, 'B');

    // Ajusta o número de iterações para obter precisão sem demorar muito em mensagens gigantes
    const int iteracoes = (volume < 1048576) ? 1000 : 50;

    MPI_Barrier(MPI_COMM_WORLD);

    const double tempo_inicio_rede = MPI_Wtime();

    for (int i = 0; i < iteracoes; i++) {
        if (meu_rank == 0) {
            MPI_Send(send_buf.data(), static_cast<int>(volume), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(recv_buf.data(), static_cast<int>(volume), MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (meu_rank == 1) {
            MPI_Recv(recv_buf.data(), static_cast<int>(volume), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(send_buf.data(), static_cast<int>(volume), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        }
    }

    const double tempo_fim_rede = MPI_Wtime();

    // Teste isolado de leitura/escrita de memória
    const std::size_t pool_size = 256ull * 1024ull * 1024ull; // 256 MB
    std::vector<char> pool_send;
    std::vector<char> pool_recv;
    std::size_t offset = 0;

    if (meu_rank == 0) {
        pool_send.resize(pool_size, 'A');
        pool_recv.resize(pool_size, 'B');
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const double tempo_inicio_mem = MPI_Wtime();

    if (meu_rank == 0) {
        for (int i = 0; i < iteracoes; i++) {
            if (offset + volume > pool_size) {
                offset = 0;
            }

            if (volume > 0) {
                std::memcpy(pool_recv.data() + offset, pool_send.data() + offset, volume);
            }

            #if defined(__GNUC__) || defined(__clang__)
            asm volatile("" ::: "memory");
            #endif

            offset += volume;
        }
    }

    const double tempo_fim_mem = MPI_Wtime();

    if (meu_rank == 0) {
        // Tempo de rede: apenas ida
        const double tempo_total_rede = tempo_fim_rede - tempo_inicio_rede;
        const double rtt_medio = tempo_total_rede / static_cast<double>(iteracoes);
        resultado.tempo_rede = rtt_medio / 2.0;

        // Tempo de memória: cópia local
        const double tempo_total_mem = tempo_fim_mem - tempo_inicio_mem;
        resultado.tempo_mem = tempo_total_mem / static_cast<double>(iteracoes);
    }

    return resultado;
}

// ============================================================================
// 3. FLUXO PRINCIPAL DO PROGRAMA MPI
// ============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int meu_rank = 0;
    int total_processos = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &meu_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processos);

    if (total_processos < 2) {
        if (meu_rank == 0) {
            std::cerr << "Erro: Este profiling de rede requer no mínimo 2 processos MPI (mpirun -np 2)." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::vector<std::size_t> vetor_de_testes;
    int tamanho_do_vetor = 0;

    if (meu_rank == 0) {
        std::cout << "Iniciando profiling para redes TCP/IP..." << std::endl;
        vetor_de_testes = Gerar_Vetor_Volumes();
        tamanho_do_vetor = static_cast<int>(vetor_de_testes.size());
    }

    MPI_Bcast(&tamanho_do_vetor, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (meu_rank != 0) {
        vetor_de_testes.resize(static_cast<std::size_t>(tamanho_do_vetor));
    }

    MPI_Bcast(
        vetor_de_testes.data(),
        tamanho_do_vetor,
        MPI_UNSIGNED_LONG_LONG,
        0,
        MPI_COMM_WORLD
    );

    std::vector<DadoBruto> dados_brutos;
    dados_brutos.reserve(static_cast<std::size_t>(tamanho_do_vetor));

    // Execução real do profiling
    for (std::size_t volume_atual : vetor_de_testes) {
        dados_brutos.push_back(Executar_Ping_Pong_E_Medir_Tempo(volume_atual, meu_rank));
    }

    // Escrita no arquivo: apenas rank 0
    if (meu_rank == 0) {
        std::ofstream arquivo_saida("profiling_results.txt");
        if (!arquivo_saida.is_open()) {
            std::cerr << "Erro ao criar 'profiling_results.txt'." << std::endl;
            MPI_Finalize();
            return 1;
        }

        arquivo_saida << "Max_Volume_Bytes\tm_TempoPorByte\tb_Latencia\n";

        if (dados_brutos.size() >= 2) {
            for (std::size_t i = 0; i + 1 < dados_brutos.size(); ++i) {
                const auto& p1 = dados_brutos[i];
                const auto& p2 = dados_brutos[i + 1];

                const double t1 = p1.tempo_rede + p1.tempo_mem;
                const double t2 = p2.tempo_rede + p2.tempo_mem;

                double m = 0.0;
                const double delta_v = static_cast<double>(p2.volume - p1.volume);

                if (delta_v > 0.0) {
                    m = (t2 - t1) / delta_v;
                    if (m < 0.0) {
                        m = 0.0;
                    }
                }

                const double b = t1 - m * static_cast<double>(p1.volume);

                arquivo_saida << p2.volume << "\t"
                              << m << "\t"
                              << b << "\n";
            }
        }

        arquivo_saida.close();
        std::cout << "\nProfiling concluído! Segmentos lineares salvos em 'profiling_results.txt'." << std::endl;
    }

    MPI_Finalize();
    return 0;
}