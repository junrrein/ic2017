#include "construir_tuplas.cpp"
#include "../../config.hpp"

int main()
{
    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    const string rutaVentas = rutaBase + "Ventas.csv";
    const string rutaExportaciones = rutaBase + "Exportaciones.csv";
    const string rutaImportaciones = rutaBase + "Importaciones.csv";

    vec ventas;
    vec exportaciones;
    vec importaciones;
    ventas.load(rutaVentas);
    exportaciones.load(rutaExportaciones);
    importaciones.load(rutaImportaciones);

    mat tuplas = agruparEntradas(6,
                                 ventas,
                                 12,
                                 exportaciones,
                                 12,
                                 importaciones,
                                 12);

    return 0;
}
