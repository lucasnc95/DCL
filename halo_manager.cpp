#include "comm/halo_manager.hpp"
#include <vector>
#include <mpi.h>

namespace dcl {

void HaloManager::sync_halos_transparent(int buffer_id, int radius, const Partition& part, DeviceManager& dev_mgr) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Identificação de vizinhos (Ref: RecuperarPosicaoHistograma)
    int left_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    size_t element_size = 4; // Assumindo float (elementSize)
    size_t halo_bytes = radius * element_size;

    // Buffers de Host para o packing (Ref: sendBuff e recBuff)
    std::vector<char> send_left(halo_bytes), send_right(halo_bytes);
    std::vector<char> recv_left(halo_bytes), recv_right(halo_bytes);

    // Itera pelos dispositivos locais do rank
    for (int i = 0; i < dev_mgr.get_local_count(); ++i) {
        Device& dev = dev_mgr.get_device(i);

        // 1. Pack: Lê as bordas do dispositivo (GPU/CPU) para o Host
        // Borda Esquerda (para enviar ao vizinho da esquerda)
        size_t left_border_off = 0; 
        // Borda Direita (para enviar ao vizinho da direita)
        size_t right_border_off = (part.length - radius) * element_size;

        clEnqueueReadBuffer(dev.dataQueue, dev.memoryObjects[buffer_id], CL_TRUE, 
                            left_border_off, halo_bytes, send_left.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(dev.dataQueue, dev.memoryObjects[buffer_id], CL_TRUE, 
                            right_border_off, halo_bytes, send_right.data(), 0, nullptr, nullptr);
        
        clFinish(dev.dataQueue); // Garante sincronização para profiling
    }

    // 2. Exchange: Comunicação MPI não-bloqueante
    MPI_Request reqs[4];
    // Envia direita, recebe do vizinho da direita (halo direito)
    MPI_Isend(send_right.data(), halo_bytes, MPI_BYTE, right_neighbor, 101, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(recv_right.data(), halo_bytes, MPI_BYTE, right_neighbor, 102, MPI_COMM_WORLD, &reqs[1]);
    
    // Envia esquerda, recebe do vizinho da esquerda (halo esquerdo)
    MPI_Isend(send_left.data(), halo_bytes, MPI_BYTE, left_neighbor, 102, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(recv_left.data(), halo_bytes, MPI_BYTE, left_neighbor, 101, MPI_COMM_WORLD, &reqs[3]);

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // 3. Unpack: Escreve os Halos recebidos nas Ghost Cells do dispositivo
    for (int i = 0; i < dev_mgr.get_local_count(); ++i) {
        Device& dev = dev_mgr.get_device(i);

        // Halo esquerdo (recebido do vizinho anterior)
        if (left_neighbor != MPI_PROC_NULL) {
            // No Stencil 1D, o halo esquerdo costuma ser colocado antes do dado útil
            // Para simplicidade deste teste, assumimos que o buffer tem espaço extra (padding)
            // ou usamos o offset designado pela biblioteca.
            clEnqueueWriteBuffer(dev.dataQueue, dev.memoryObjects[buffer_id], CL_TRUE, 
                                 0, halo_bytes, recv_left.data(), 0, nullptr, nullptr);
        }
        
        // Halo direito (recebido do vizinho posterior)
        if (right_neighbor != MPI_PROC_NULL) {
            size_t right_halo_off = part.length * element_size; 
            clEnqueueWriteBuffer(dev.dataQueue, dev.memoryObjects[buffer_id], CL_TRUE, 
                                 right_halo_off, halo_bytes, recv_right.data(), 0, nullptr, nullptr);
        }
        clFlush(dev.dataQueue);
    }
}

}