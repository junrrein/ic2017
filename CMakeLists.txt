cmake_minimum_required(VERSION 3.3)

project(ic-template)

# Requisitos para usar la plantilla
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
find_package (OpenMP REQUIRED)
find_package (Gnuplot REQUIRED)

# Descargar y compilar dependencias externas
add_subdirectory (external)
include_directories (SYSTEM
	${openblas_INCLUDE_DIRS}
	${arma_INCLUDE_DIRS}
	${gpio_INCLUDE_DIRS}
)

# Banderas de compilación para los ejercicios y ejemplos
set (CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -pedantic -march=native -fopenmp")

# Ejercicios y ejemplos
add_subdirectory (ejemplos)
