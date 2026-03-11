__kernel void stencil_1d(
    __global float* data, 
    __global float* result, 
    const int radius,
    const int total_elements) 
{
    // get_global_id(0) retornará o índice global baseado no offset 
    // configurado no clEnqueueNDRangeKernel pela biblioteca
    int gid = get_global_id(0);

    // Evita processar fora dos limites globais (considerando o halo)
    if (gid >= radius && gid < (total_elements - radius)) {
        float sum = 0.0f;
        
        // Aplica o stencil (ex: média simples dos vizinhos)
        for (int i = -radius; i <= radius; i++) {
            sum += data[gid + i];
        }
        
        result[gid] = sum / (2 * radius + 1);
    }
}