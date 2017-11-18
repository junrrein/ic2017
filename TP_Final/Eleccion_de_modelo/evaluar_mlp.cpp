#include "construir_tuplas.cpp"
#include "../../guia2/mlp_salida_lineal.cpp"
#include "../../config.hpp"

using namespace ic;

// Evalúa una cierta configuración de un MLP 10 veces.
// Devuelve el promedio del error cuadrático promedio
// en una partición de evaluación.
// De los datos que se reciben, se descarta el último 10%
// (se dejan para pruebas posteriores), y del 90% restante
// se aparta el 20% para hacer la evaluación.
double evaluarMLP(const ParametrosMulticapa& parametros,
                  const vector<string>& rutasSeriesEntrada,
                  const string& rutaSerieSalida,
                  bool agregarIndice = false);

int main()
{
    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    const string rutaVentas = rutaBase + "Ventas.csv";
    const string rutaExportaciones = rutaBase + "Exportaciones.csv";
    const string rutaImportaciones = rutaBase + "Importaciones.csv";

    ParametrosMulticapa parametros;
    parametros.estructuraRed = {7, 6};
    parametros.nEpocas = 3000;
    parametros.tasaAprendizaje = 0.00075;
    parametros.inercia = 0.2;
    parametros.toleranciaError = 0.1;

    double errorPromedio = evaluarMLP(parametros, {rutaVentas}, rutaVentas);

    cout << "El promedio del error cuadrático promedio es: " << errorPromedio << endl;

    return 0;
}

double evaluarMLP(const ParametrosMulticapa& parametros,
                  const vector<string>& rutasSeriesEntrada,
                  const string& rutaSerieSalida,
                  bool agregarIndice)
{
    mat tuplasEntrada, tuplasSalida;
    tie(tuplasEntrada, tuplasSalida) = cargarTuplas(rutasSeriesEntrada,
                                                    rutaSerieSalida,
                                                    parametros.estructuraRed(0),
                                                    parametros.estructuraRed(1));

    tuplasEntrada = tuplasEntrada.rows(0, tuplasEntrada.n_rows * 0.9);
    tuplasSalida = tuplasSalida.rows(0, tuplasSalida.n_rows * 0.9);

    if (agregarIndice)
        tuplasEntrada = agregarIndiceTemporal(tuplasEntrada);

    const int nDatosEntrenamiento = tuplasEntrada.n_rows * 0.8;
    const int nDatosPrueba = tuplasEntrada.n_rows - nDatosEntrenamiento;

    const mat datosEntrenamiento = join_horiz(tuplasEntrada.head_rows(nDatosEntrenamiento),
                                              tuplasSalida.head_rows(nDatosEntrenamiento));

    vec erroresPromedio(10);

#pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
        vector<mat> pesos;
        tie(pesos, ignore, ignore) = entrenarMulticapa(parametros.estructuraRed,
                                                       datosEntrenamiento,
                                                       parametros.nEpocas,
                                                       parametros.tasaAprendizaje,
                                                       parametros.inercia,
                                                       parametros.toleranciaError,
                                                       true);

        double errorTotal = errorCuadraticoMulticapa(pesos,
                                                     tuplasEntrada.tail_rows(nDatosPrueba),
                                                     tuplasSalida.tail_rows(nDatosPrueba));
        double errorPromedio = errorTotal / nDatosPrueba;
        erroresPromedio(i) = errorPromedio;
    }

    return mean(erroresPromedio);
}
