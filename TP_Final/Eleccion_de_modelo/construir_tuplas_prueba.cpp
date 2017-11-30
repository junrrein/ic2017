#include "construir_tuplas.cpp"
#include "../../config.hpp"

int main()
{
    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    const string rutaVentas = rutaBase + "Ventas.csv";
    const string rutaExportaciones = rutaBase + "Exportaciones.csv";
    const string rutaImportaciones = rutaBase + "Importaciones.csv";

    Particion particion = cargarTuplas({rutaVentas},
                                       rutaVentas,
                                       8,
                                       3);

    vec ventas;
    ventas.load(rutaVentas);

    for (int i = 0; i < 8; ++i) {
        particion.entrenamiento.tuplasEntrada.col(i) = desnormalizar(ventas, particion.entrenamiento.tuplasEntrada.col(i));
        particion.evaluacion.tuplasEntrada.col(i) = desnormalizar(ventas, particion.evaluacion.tuplasEntrada.col(i));
        particion.prueba.tuplasEntrada.col(i) = desnormalizar(ventas, particion.prueba.tuplasEntrada.col(i));
    }

    for (int i = 0; i < 3; ++i) {
        particion.entrenamiento.tuplasSalida.col(i) = desnormalizar(ventas, particion.entrenamiento.tuplasSalida.col(i));
        particion.evaluacion.tuplasSalida.col(i) = desnormalizar(ventas, particion.evaluacion.tuplasSalida.col(i));
        particion.prueba.tuplasSalida.col(i) = desnormalizar(ventas, particion.prueba.tuplasSalida.col(i));
    }

    return 0;
}
