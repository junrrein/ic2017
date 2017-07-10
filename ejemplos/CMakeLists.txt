set (PREFIJO ejemplos)

add_executable (${PREFIJO}-graficos EXCLUDE_FROM_ALL graficos.cpp)
target_link_libraries (${PREFIJO}-graficos ${IC_LIBS})

add_executable (${PREFIJO}-omp EXCLUDE_FROM_ALL omp.cpp)
target_link_libraries (${PREFIJO}-omp ${IC_LIBS})

add_custom_target (ejemplos DEPENDS ${PREFIJO}-graficos ${PREFIJO}-omp)
