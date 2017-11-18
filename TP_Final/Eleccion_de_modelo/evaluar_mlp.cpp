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
    arma_rng::set_seed_random();

    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    const string rutaVentas = rutaBase + "Ventas.csv";
    const string rutaExportaciones = rutaBase + "Exportaciones.csv";
    const string rutaImportaciones = rutaBase + "Importaciones.csv";
    const vector<string> rutas = {rutaVentas,
                                  rutaExportaciones,
                                  rutaImportaciones};
    const vector<vector<string>> subconjuntosRutas = {{rutaVentas},
                                                      {rutaExportaciones},
                                                      {rutaImportaciones},
                                                      {rutaVentas, rutaExportaciones},
                                                      {rutaVentas, rutaImportaciones},
                                                      {rutaImportaciones, rutaExportaciones},
                                                      rutas};
    const vector<int> neuronasPrimerCapa = {7, 9, 11, 14, 17, 20, 24};

    ParametrosMulticapa parametros;
    parametros.estructuraRed = {7, 6};
    parametros.nEpocas = 3000;
    parametros.tasaAprendizaje = 0.00075;
    parametros.inercia = 0.2;
    parametros.toleranciaError = 10;
    ParametrosMulticapa mejoresParametros = parametros;
    double mejorError = numeric_limits<double>::max();
    vector<string> mejorSubconjunto;

    for (const vector<string>& rutasEntradas : subconjuntosRutas) {
        for (int cantidadNeuronas : neuronasPrimerCapa) {
            parametros.estructuraRed = {double(cantidadNeuronas), 6};

            double promedioErrorPromedio = evaluarMLP(parametros, rutasEntradas, rutaVentas);

            cout << "El promedio del error cuadrático promedio es: " << promedioErrorPromedio << endl;

            if (promedioErrorPromedio < mejorError) {
                mejoresParametros = parametros;
                mejorError = promedioErrorPromedio;
                mejorSubconjunto = rutasEntradas;
            }
        }
    }

    cout << "Mejor subconjunto: " << endl;
    for (const string& ruta : mejorSubconjunto)
        cout << ruta << endl;
    cout << "Mejor estructura:\n"
         << mejoresParametros.estructuraRed
         << "Mejor error: " << mejorError << endl;

    return 0;
}

double evaluarMLP(const ParametrosMulticapa& parametros,
                  const vector<string>& rutasSeriesEntrada,
                  const string& rutaSerieSalida,
                  bool agregarIndice)
{
    Particion particion = cargarTuplas(rutasSeriesEntrada,
                                       rutaSerieSalida,
                                       parametros.estructuraRed(0),
                                       parametros.estructuraRed(1));

    if (agregarIndice) {
        particion.entrenamiento.tuplasEntrada = agregarIndiceTemporal(particion.entrenamiento.tuplasEntrada);
        particion.evaluacion.tuplasEntrada = agregarIndiceTemporal(particion.evaluacion.tuplasEntrada);
    }

    const mat datosEntrenamiento = join_horiz(particion.entrenamiento.tuplasEntrada,
                                              particion.entrenamiento.tuplasSalida);
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
                                                     particion.evaluacion.tuplasEntrada,
                                                     particion.evaluacion.tuplasSalida);
        double errorPromedio = errorTotal / particion.evaluacion.tuplasEntrada.n_rows;
        erroresPromedio(i) = errorPromedio;
    }

    return mean(erroresPromedio);
}
