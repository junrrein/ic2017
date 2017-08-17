agregar_ejercicio (graficos)
agregar_ejercicio (omp)
agregar_ejercicio (variosArchivos)
agregar_fuentes (variosArchivos "funcion1.cpp funcion2.cpp")
agregar_datos ("sin.txt")

set_target_properties(ejemplos-graficos ejemplos-omp ejemplos-variosArchivos
    PROPERTIES EXCLUDE_FROM_ALL 1)
add_custom_target(ejemplos DEPENDS ejemplos-graficos ejemplos-omp ejemplos-variosArchivos)
