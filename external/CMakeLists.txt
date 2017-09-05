cmake_minimum_required(VERSION 3.3)

include (ExternalProject)

###########################
#	OpenBLAS
###########################

if (WIN32)
    # TODO: Usar -no-start
    # https://github.com/Alexpux/MSYS2-packages/blob/master/filesystem/msys2_shell.cmd#L164
    set (openblas_BUILD_COMMAND ${MSYS_PATH}/msys2_shell.cmd -here -mingw32 -c "make USE_OPENMP=1")
else ()
    set (openblas_BUILD_COMMAND make USE_OPENMP=1)
endif ()

ExternalProject_Add (openblas
    EXCLUDE_FROM_ALL 1

    URL http://github.com/xianyi/OpenBLAS/archive/v0.2.19.tar.gz

    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openblas"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${openblas_BUILD_COMMAND}
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
)

set (openblas_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/openblas" PARENT_SCOPE)
set (openblas_LIBS
    "${CMAKE_CURRENT_SOURCE_DIR}/openblas/${CMAKE_SHARED_LIBRARY_PREFIX}openblas${CMAKE_SHARED_LIBRARY_SUFFIX}"
	PARENT_SCOPE
)

###########################
#	HDF5
###########################

#TODO: Hacer andar esto en Windows

if (WIN32)
    # TODO: Usar -no-start
    # https://github.com/Alexpux/MSYS2-packages/blob/master/filesystem/msys2_shell.cmd#L164
    #set (hdf5_BUILD_COMMAND ${MSYS_PATH}/msys2_shell.cmd -here -mingw32 -c "make USE_OPENMP=1")
else ()
    #set (hdf5_BUILD_COMMAND sh build-unix.sh)
endif ()

set (hdf5_CONFIGURE_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/hdf5/configure --prefix=${CMAKE_CURRENT_SOURCE_DIR}/hdf5)
set (hdf5_INSTALL_COMMAND "make install")

ExternalProject_Add (hdf5
    EXCLUDE_FROM_ALL 1

    URL https://support.hdfgroup.org/ftp/HDF5/current18/src/hdf5-1.8.19.tar.gz

    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hdf5"
    CONFIGURE_COMMAND ${hdf5_CONFIGURE_COMMAND}
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    # CMake corre "make install" solo
)

set (hdf5_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/hdf5/include" PARENT_SCOPE)
set (hdf5_LIBS
    "${CMAKE_CURRENT_SOURCE_DIR}/hdf5/lib/${CMAKE_SHARED_LIBRARY_PREFIX}hdf5${CMAKE_SHARED_LIBRARY_SUFFIX}"
	PARENT_SCOPE
)

###########################
#	Armadillo
###########################

ExternalProject_Add (armadillo
    EXCLUDE_FROM_ALL 1

    URL http://sourceforge.net/projects/arma/files/armadillo-7.950.1.tar.xz

    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/armadillo"
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    # No construir la biblioteca wrapper; no la vamos a usar.
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

set (arma_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/armadillo/include" PARENT_SCOPE)

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

    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gnuplot-iostream"
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

set (gpio_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/gnuplot-iostream" PARENT_SCOPE)

#################################################################
#################################################################
#################################################################

# Objetivo de CMake para construir todas estas cosas de una
add_custom_target (external DEPENDS openblas hdf5 armadillo gnuplot-iostream)

