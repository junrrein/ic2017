cmake_minimum_required(VERSION 3.3)

project(ic-template)

# Requisitos para usar la plantilla
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
find_package (OpenMP REQUIRED)
find_package (Gnuplot REQUIRED)

# Dependencias externas
add_subdirectory (external)

# Banderas de compilación para los ejercicios y ejemplos
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
add_definitions (-DARMA_DONT_USE_WRAPPER)
include_directories (SYSTEM
	${openblas_INCLUDE_DIRS}
	${arma_INCLUDE_DIRS}
	${gpio_INCLUDE_DIRS}
)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -pedantic -march=native")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++14 /W4")
endif()

# Bibliotecas a usar en los ejercicios y ejemplos
set (IC_LIBS ${openblas_LIBS} Boost::filesystem Boost::iostreams)

# Ejemplos
add_subdirectory (ejemplos)

# Ejercicios
add_subdirectory (guia1)

