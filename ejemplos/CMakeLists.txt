set (PREFIJO ejemplos)

agregar_ejercicio (graficos)
agregar_ejercicio (omp)

add_custom_target (ejemplos DEPENDS ${PREFIJO}-graficos ${PREFIJO}-omp)
