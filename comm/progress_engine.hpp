#include <CL/cl.h>
#include <mpi.h>


namespace dcl {
    class ProgressEngine {
    public:
        void start();
        void stop();
        void add_request(MPI_Request req); // Registra Isend/Irecv
        void add_event(cl_event ev);       // Registra eventos OpenCL
        void wait_all();                   // Sincronização final
    };
}

