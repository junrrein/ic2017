cmake_minimum_required(VERSION 3.3)

project(ic-template)

# Requisitos para usar la plantilla
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(Fortran)
find_package (OpenMP REQUIRED)
find_package (Gnuplot REQUIRED)

if (WIN32)
    if (EXISTS C:/msys64)
        set (MSYS_PATH C:/msys64)
    elseif (EXISTS C:/msys32)
        set (MSYS_PATH C:/msys32)
    else ()
        message (FATAL_ERROR "No se pudo encontrar la instalaciòn de MSYS2")
    endif ()

    set (MSYS_LIBRARY_PATH ${MSYS_PATH}/mingw32/bin)
endif ()

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
if (WIN32)
    set (GCC_LIB ${MSYS_LIBRARY_PATH}/libgcc_s_dw2-1.dll)
    set (gomp_LIB ${MSYS_LIBRARY_PATH}/libgomp-1.dll)
    set (Fortran_LIB ${MSYS_LIBRARY_PATH}/libgfortran-3.dll)
    set (WinPThread_LIB ${MSYS_LIBRARY_PATH}/libwinpthread-1.dll)
    set (QuadMath_LIB ${MSYS_LIBRARY_PATH}/libquadmath-0.dll)
    # OpenBLAS se copia separadamente
    set (COPIAR_LIBS ${GCC_LIB} ${gomp_LIB} ${Fortran_LIB} ${WinPThread_LIB} ${QuadMath_LIB})
endif ()

# Función para agregar una nueva guía
function (agregar_guia)
    if (WIN32)
        file (COPY ${COPIAR_LIBS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        if (EXISTS ${openblas_LIBS})
            # Se copia en una operación separada porque no existe inicialmente
            file (COPY ${openblas_LIBS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        endif ()
    endif ()
endfunction ()

# Función para agregar un nuevo ejercicio
function (agregar_ejercicio nombre)
    get_filename_component(PREFIJO ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    add_executable (${PREFIJO}-${nombre} ${nombre}.cpp)
    target_link_libraries (${PREFIJO}-${nombre} ${IC_LIBS})
    set_target_properties(${PREFIJO}-${nombre} PROPERTIES OUTPUT_NAME ${nombre})
endfunction (agregar_ejercicio)

# Ejemplos
add_subdirectory (ejemplos)

# Ejercicios
add_subdirectory (guia1)

