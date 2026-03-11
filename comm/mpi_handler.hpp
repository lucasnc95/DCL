#ifndef DCL_MPI_HANDLER_HPP
#define DCL_MPI_HANDLER_HPP

#include <mpi.h>
#include <vector>

namespace dcl {

class MPIHandler {
public:
    MPIHandler(int &argc, char** &argv) {
        int provided;
        // Essencial para a futura Thread de Progresso
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    }

    int rank() const { return m_rank; }
    int size() const { return m_size; }

    // Sincronização global para balanceamento
    void barrier() { MPI_Barrier(MPI_COMM_WORLD); }

private:
    int m_rank, m_size;
};

}
#endif