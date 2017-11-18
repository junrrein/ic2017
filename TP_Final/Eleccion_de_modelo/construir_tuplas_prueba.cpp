#include "construir_tuplas.cpp"
#include "../../config.hpp"

int main()
{
    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    const string rutaVentas = rutaBase + "Ventas.csv";
    const string rutaExportaciones = rutaBase + "Exportaciones.csv";
    const string rutaImportaciones = rutaBase + "Importaciones.csv";

    mat tuplas = cargarTuplas({rutaVentas, rutaExportaciones, rutaImportaciones},
                              rutaVentas,
                              12,
                              6);
    mat tuplasConIndice = agregarIndiceTemporal(tuplas);

    return 0;
}
