cmake_minimum_required(VERSION 3.3)

include (ExternalProject)

###########################
#	OpenBLAS
###########################

ExternalProject_Add (openblas
    EXCLUDE_FROM_ALL 1

    URL http://github.com/xianyi/OpenBLAS/archive/v0.2.19.tar.gz

    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    SOURCE_DIR "${CMAKE_BINARY_DIR}/external/openblas"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make USE_OPENMP=1
    BUILD_IN_SOURCE 1
    USES_TERMINAL_BUILD 0
    INSTALL_COMMAND ""
)

set (openblas_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/external/openblas" PARENT_SCOPE)
set (openblas_LIBS
	"${CMAKE_BINARY_DIR}/external/openblas/${CMAKE_SHARED_LIBRARY_PREFIX}openblas${CMAKE_SHARED_LIBRARY_SUFFIX}"
	PARENT_SCOPE
)

###########################
#	Armadillo
###########################

ExternalProject_Add (armadillo
    EXCLUDE_FROM_ALL 1

    URL http://sourceforge.net/projects/arma/files/armadillo-7.950.1.tar.xz

	SOURCE_DIR "${CMAKE_BINARY_DIR}/external/armadillo"
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    # Don't build the wrapper library; we won't use it.
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

set (arma_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/external/armadillo/include" PARENT_SCOPE)

###########################
#	Boost
###########################

set (BOOST_DOWNLOAD_TO_BINARY_DIR ON)
add_subdirectory (boost-cmake)

###########################
#	gnuplot-iostream
###########################

ExternalProject_Add (gnuplot-iostream
    EXCLUDE_FROM_ALL 1

    URL https://github.com/dstahlke/gnuplot-iostream/archive/master.zip

	SOURCE_DIR "${CMAKE_BINARY_DIR}/external/gnuplot-iostream"
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

set (gpio_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/external/gnuplot-iostream" PARENT_SCOPE)

#################################################################
#################################################################
#################################################################

# Objetivo de CMake para construir todas estas cosas de una
add_custom_target (external DEPENDS openblas armadillo gnuplot-iostream)

