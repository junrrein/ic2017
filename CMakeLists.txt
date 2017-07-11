cmake_minimum_required(VERSION 3.3)

project(ic-template)

# Requisitos para usar la plantilla
set (CMAKE_CXX_STANDARD 11)
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
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -march=native")

# Bibliotecas a usar en los ejercicios y ejemplos
set (IC_LIBS ${openblas_LIBS} Boost::filesystem Boost::iostreams)

# Función para agregar un nuevo ejercicio
function (agregar_ejercicio nombre)
    add_executable (${PREFIJO}-${nombre} ${nombre}.cpp)
    target_link_libraries (${PREFIJO}-${nombre} ${IC_LIBS})
    set_target_properties(${PREFIJO}-${nombre} PROPERTIES OUTPUT_NAME ${nombre})
endfunction (agregar_ejercicio)

# Ejemplos
add_subdirectory (ejemplos)

# Ejercicios
add_subdirectory (guia1)

