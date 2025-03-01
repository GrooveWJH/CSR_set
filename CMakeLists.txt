cmake_minimum_required(VERSION 3.18)
project(CSR_SET LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_language(CUDA)

set(HELPER_SOURCES helper.cpp helper.h)

# add_executable(csr_scaler csr_scaler.cu ${HELPER_SOURCES})
# set_target_properties(csr_scaler PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_RESOLVE_DEVICE_SYMBOLS ON
# )
# target_link_libraries(csr_scaler PRIVATE CUDA::cudart)

# add_executable(csr_vector csr_vector.cu ${HELPER_SOURCES})
# set_target_properties(csr_vector PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_RESOLVE_DEVICE_SYMBOLS ON
# )target_link_libraries(csr_vector PRIVATE CUDA::cudart)

add_executable(csr_adaptive csr_adaptive.cu ${HELPER_SOURCES})
set_target_properties(csr_adaptive PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(csr_adaptive PRIVATE CUDA::cudart)

find_package(CUDAToolkit REQUIRED)


