cmake_minimum_required(VERSION 3.3)

include (ExternalProject)

###########################
#	OpenBLAS
###########################

ExternalProject_Add (openblas
    EXCLUDE_FROM_ALL 1

    URL http://github.com/xianyi/OpenBLAS/archive/v0.2.19.tar.gz

    PREFIX openblas

    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/openblas"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make USE_OPENMP=1
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
)

set (openblas_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/external/openblas" PARENT_SCOPE)
set (openblas_LIBS "${CMAKE_SOURCE_DIR}/external/openblas/libopenblas.so" PARENT_SCOPE)

###########################
#	Armadillo
###########################

ExternalProject_Add (armadillo
    EXCLUDE_FROM_ALL 1

    URL http://sourceforge.net/projects/arma/files/armadillo-7.950.1.tar.xz

    PREFIX armadillo

	SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/armadillo"
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    # Don't build the wrapper library; we won't use it.
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

add_definitions (-DARMA_DONT_USE_WRAPPER)
set (arma_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/external/armadillo/include" PARENT_SCOPE)

###########################
#	Boost
###########################

set (BOOST_URL https://github.com/boostorg/boost/archive/boost-1.64.0.tar.gz)
add_subdirectory (boost-cmake)

###########################
#	gnuplot-iostream
###########################

ExternalProject_Add (gnuplot-iostream
    EXCLUDE_FROM_ALL 1

    URL https://github.com/dstahlke/gnuplot-iostream/archive/master.zip

    PREFIX gnuplot-iostream

	SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/gnuplot-iostream"
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

set (gpio_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/external/gnuplot-iostream" PARENT_SCOPE)

#################################################################
#################################################################
#################################################################

# Objetivo de CMake para construir todas estas cosas de una
add_custom_target (external DEPENDS openblas armadillo gnuplot-iostream)

