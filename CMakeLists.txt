cmake_minimum_required(VERSION 3.3)

# Lo siguiente evita un error raro en algunos sistemas.
# El error es:
# ic/external/boost-cmake/cmake/Modules/AddBoostLib.cmake:7: error:
# Target "Boost_mpi" links to item " -Wl,-rpath -Wl,/usr/lib/openmpi/lib
# -Wl,--enable-new-dtags" which has leading or trailing whitespace.
# This is now an error according to policy CMP0004. external/boost-cmake/libs/mpi.cmake:7
# (_add_boost_lib) external/boost-cmake/CMakeLists.txt:62 (include)
cmake_policy (SET CMP0004 OLD)

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
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wshadow -pedantic -march=native")

# Bibliotecas a usar en los ejercicios y ejemplos
set (IC_LIBS ${openblas_LIBS} Boost::filesystem Boost::iostreams)
if (WIN32)
    file (GLOB GCC_LIB ${MSYS_LIBRARY_PATH}/libgcc*.dll)
    file (GLOB gomp_LIB ${MSYS_LIBRARY_PATH}/libgomp*.dll)
    file (GLOB Fortran_LIB ${MSYS_LIBRARY_PATH}/libgfortran*.dll)
    file (GLOB WinPThread_LIB ${MSYS_LIBRARY_PATH}/libwinpthread*.dll)
    file (GLOB QuadMath_LIB ${MSYS_LIBRARY_PATH}/libquadmath*.dll)
    set (COPIAR_LIBS ${openblas_LIBS} ${GCC_LIB} ${gomp_LIB} ${Fortran_LIB} ${WinPThread_LIB} ${QuadMath_LIB})
endif ()

# Función para agregar un nuevo ejercicio
function (agregar_ejercicio nombre_ejercicio)
    get_filename_component(PREFIJO ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    add_executable (${PREFIJO}-${nombre_ejercicio} ${nombre_ejercicio}.cpp)
    target_link_libraries (${PREFIJO}-${nombre_ejercicio} ${IC_LIBS})
    set_target_properties(${PREFIJO}-${nombre_ejercicio} PROPERTIES OUTPUT_NAME ${nombre_ejercicio})

    if (WIN32)
        add_custom_command (
            TARGET ${PREFIJO}-${nombre_ejercicio}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${COPIAR_LIBS} ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif ()
endfunction (agregar_ejercicio)

# Función para agregar fuentes a un ejercicio existente
function (agregar_fuentes nombre_ejercicio string_archivos)
    get_filename_component (PREFIJO ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    separate_arguments (lista_archivos UNIX_COMMAND ${string_archivos})
    target_sources (${PREFIJO}-${nombre_ejercicio} PUBLIC ${lista_archivos})
endfunction ()

# Configurar el archivo que va a tener la referencia al directorio fuente del proyecto
configure_file(config.hpp.cmake ${CMAKE_SOURCE_DIR}/config.hpp)

# Ejemplos
add_subdirectory (ejemplos)

# Ejercicios
add_subdirectory (guia1)

