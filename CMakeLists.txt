cmake_minimum_required(VERSION 3.3)

project(ic-template)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
find_package (OpenMP REQUIRED)
find_package (Gnuplot REQUIRED)
find_package (Boost REQUIRED COMPONENTS system iostreams filesystem)

include (external/CMakeLists.txt)

set (CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -pedantic \
    -march=native -fopenmp \
    -lboost_system -lboost_iostreams -lboost_filesystem")

add_executable (graficos "ejemplos/graficos.cpp")
target_link_libraries (graficos ${openblas_LIBS})

add_executable (omp "ejemplos/omp.cpp")
target_link_libraries (graficos ${openblas_LIBS})
