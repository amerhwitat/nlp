cmake_minimum_required(VERSION 3.20)
project(ChimeraFlashTool LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(flashtool_core STATIC
    ../core/src/flashtool.cpp
    ../core/src/flash_features.cpp
)
target_include_directories(flashtool_core PUBLIC ../core/include)

add_library(flashtool_c STATIC ../c/flashtool.c)
target_include_directories(flashtool_c PUBLIC ../core/include)
target_link_libraries(flashtool_c PUBLIC flashtool_core)

add_executable(flashtool_core_smoke ../tests/core_smoke.cpp)
target_include_directories(flashtool_core_smoke PRIVATE ../core/include)
target_link_libraries(flashtool_core_smoke PRIVATE flashtool_core)
