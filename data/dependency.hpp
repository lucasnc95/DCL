#ifndef DCL_DEPENDENCY_HPP
#define DCL_DEPENDENCY_HPP

#include "../core/types.hpp"

namespace dcl {
    struct DataDependency {
        int buffer_id;
        AccessMode mode; // Alterado de 'type' para 'mode' para evitar conflito com palavras reservadas
        int halo_radius = 0;

        static DataDependency stencil(int id, int radius) {
            return {id, AccessMode::NEIGHBORHOOD, radius};
        }
    };
}
#endif