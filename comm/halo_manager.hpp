#ifndef DCL_HALO_MANAGER_HPP
#define DCL_HALO_MANAGER_HPP

#include "../core/device_manager.hpp"
#include "../data/box.hpp"
#include <mpi.h>

namespace dcl {

    class HaloManager {
    public:
        /**
         * @brief Sincroniza as bordas (halos) entre processos vizinhos.
         * Substitui a antiga função Comms() da biblioteca original.
         * * @param buffer_id ID global do buffer OpenCL.
         * @param radius Raio da borda (sdSize).
         * @param part Partição (offset/length) do dispositivo atual.
         * @param dev_mgr Referência ao gerenciador para acessar as filas de dados.
         */
        void sync_halos_transparent(int buffer_id, int radius, const Partition& part, DeviceManager& dev_mgr);
    
    private:
        /**
         * @brief Extrai dados da GPU (strided ou contíguos) para o Host.
         */
        void pack_and_send(int buffer_id, const MemoryLayout& layout, int dest_rank, DeviceManager& dev_mgr);

        /**
         * @brief Recebe dados do Host e insere na GPU na posição de Halo.
         */
        void recv_and_unpack(int buffer_id, const MemoryLayout& layout, int src_rank, DeviceManager& dev_mgr);
    };

} // namespace dcl

#endif