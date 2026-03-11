#ifndef DCL_BOX_HPP
#define DCL_BOX_HPP

#include <vector>
#include <cstddef>

namespace dcl {

// Abstração para suportar domínios de 1D a 4D
struct BoxND {
    std::vector<size_t> dims; 
    size_t total_elements() const {
        size_t total = 1;
        for (auto d : dims) total *= d;
        return total;
    }
};

/**
 * Define a fatia de trabalho de cada dispositivo.
 * Substitui os arrays 'offset' e 'length' da implementação original.
 */
struct Partition {
    int global_device_id; // ID único do dispositivo no cluster
    size_t offset;        // Ponto inicial na memória global (em elementos)
    size_t length;        // Quantidade de elementos designada a este dispositivo
    
    // Opcional: Para suporte 3D/4D, armazena a forma local da fatia
    BoxND local_shape; 
};

/**
 * Define como extrair uma fatia de dados não contíguos.
 * Essencial para extrair faces laterais de volumes 3D/4D.
 */
struct MemoryLayout {
    size_t start_offset; // Índice do primeiro elemento da borda
    size_t count;        // Quantos elementos contíguos (largura da face em X)
    size_t stride;       // Quantos elementos pular para chegar à próxima linha/plano
    size_t repetitions;  // Quantas vezes repetir o salto (número de linhas/planos)
};

} // namespace dcl

#endif