#include "runtime/runtime.hpp"
#include <vector>
#include <iostream>

int main(int argc, char** argv) {
    dcl::Runtime runtime(argc, argv);
    int rank = runtime.rank();

    runtime.init_devices(dcl::DeviceTag::ALL, 2, true); 

    size_t N = 1024*2048; 
    std::vector<float> h_input(N, 1.0f);
    if (rank == 0) h_input[N/2] = 100.0f; // Pulso inicial

    // Criamos dois buffers para o Ping-Pong
    int buf_A = runtime.create_buffer(N, CL_MEM_READ_WRITE);
    int buf_B = runtime.create_buffer(N, CL_MEM_READ_WRITE);

    runtime.create_kernel("stencil_kernel.cl", "stencil_1d");

    // Upload inicial apenas para o buffer A
    runtime.write_buffer(buf_A, h_input.data(), 0, N);

    int radius = 1;
    int total = (int)N;
    int iterations = 100;

    for (int t = 0; t < iterations; ++t) {
        // Define quem é entrada e quem é saída nesta iteração
        int input = (t % 2 == 0) ? buf_A : buf_B;
        int output = (t % 2 == 0) ? buf_B : buf_A;

        // Atualiza apenas os dois primeiros argumentos (os buffers)
        runtime.set_arg(0, input);
        runtime.set_arg(1, output);
        runtime.set_arg(2, &radius, sizeof(int));
        runtime.set_arg(3, &total, sizeof(int));

        // Dependência de vizinhança para o buffer de entrada
        dcl::DataDependency dep = dcl::DataDependency::stencil(input, radius);
        
        runtime.enqueue_kernel(dep);
        
    }

    // O resultado final estará no buffer que foi o 'output' da última iteração
    int final_buf = (iterations % 2 == 0) ? buf_A : buf_B;
    std::vector<float> h_output(N);
    std::cout << "Aguardando conclusao de todos os dispositivos..." << std::endl;
    runtime.wait_all();

    runtime.gather_global(final_buf, h_output.data());

    if (rank == 0) {
        std::cout << "Resultado final no centro: " << h_output[N/2] << std::endl;
    }

    return 0;
}