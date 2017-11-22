#include "construir_tuplas.cpp"
#include "../../config.hpp"

int main()
{
    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    const string rutaVentas = rutaBase + "Ventas.csv";
    const string rutaExportaciones = rutaBase + "Exportaciones.csv";
    const string rutaImportaciones = rutaBase + "Importaciones.csv";

    Particion particion = cargarTuplas({rutaVentas, rutaExportaciones, rutaImportaciones},
                                       rutaVentas,
                                       10,
                                       6);

    return 0;
}