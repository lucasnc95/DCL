#ifndef DCL_TYPES_HPP
#define DCL_TYPES_HPP

namespace dcl {
    enum class DeviceTag { CPU, GPU, ALL };
    enum class AccessMode { ONE_TO_ONE, NEIGHBORHOOD }; // Consolidado aqui
}

#endif