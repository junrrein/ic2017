cmake_minimum_required(VERSION 2.8)
project(ic-template)

find_package (Armadillo REQUIRED)
find_package (OpenMP REQUIRED)
find_package (Gnuplot REQUIRED)
find_package (Boost REQUIRED COMPONENTS system iostreams filesystem)

add_definitions (-DARMA_DONT_USE_WRAPPER)
set (CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -pedantic \
    -march=native -fopenmp \
    -I${CMAKE_SOURCE_DIR}/OpenBLAS -L${CMAKE_SOURCE_DIR}/OpenBLAS -lopenblas_openmp \
    -lboost_system -lboost_iostreams -lboost_filesystem")

add_executable (ejemplo-simple "ejemplos/ejemplo-simple.cpp")
add_executable (ejemplo-sin-omp "ejemplos/ejemplo-sin-omp.cpp")
add_executable (ejemplo-con-omp "ejemplos/ejemplo-con-omp.cpp")
