cmake_minimum_required(VERSION 4.0)
project(cuda_hook LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(Boost 1.83.0 CONFIG REQUIRED COMPONENTS filesystem json)

add_library(cuda_hook SHARED
    csrc/hook.cpp
)

target_include_directories(cuda_hook PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuda_hook PRIVATE
    # NOTE(yongji): cuda_hook should link against
    # neither CUDA driver or CUDA runtime
    # CUDA::cuda_driver
    dl
    pthread
    Boost::filesystem
    Boost::json
)

if(DEFINED ENV{HOOK_DEBUG})
    message(STATUS "HOOK_DEBUG is defined, enabling debug mode")
    target_compile_definitions(cuda_hook PRIVATE HOOK_DEBUG)
endif()

set_target_properties(cuda_hook PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH "$ORIGIN"
    OUTPUT_NAME "cuda_hook"
    PREFIX "lib"
    SUFFIX ".so"
)

install(TARGETS cuda_hook
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
