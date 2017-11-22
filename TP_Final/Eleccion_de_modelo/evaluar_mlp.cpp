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
                  const Particion& particion);

int main()
{
    arma_rng::set_seed_random();

    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    const string rutaVentas = rutaBase + "Ventas.csv";
    const string rutaDiferencias = rutaBase + "DiferenciasRelativas.csv";
    const string rutaExportaciones = rutaBase + "Exportaciones.csv";
    const string rutaImportaciones = rutaBase + "Importaciones.csv";
    const vector<string> rutas = {rutaVentas,
                                  rutaDiferencias,
                                  rutaExportaciones,
                                  rutaImportaciones};
    const vector<vector<string>> subconjuntosRutas = subconjuntos<4>(rutas);
    vector<int> neuronasPrimerCapa(2);
    iota(neuronasPrimerCapa.begin(), neuronasPrimerCapa.end(), 7);
    vector<unsigned int> retrasosAProbar(2);
    iota(retrasosAProbar.begin(), retrasosAProbar.end(), 4);

    ParametrosMulticapa parametros;
    parametros.nEpocas = 2000;
    parametros.tasaAprendizaje = 0.00075;
    parametros.inercia = 0.2;
    parametros.toleranciaError = 10;
    ParametrosMulticapa mejoresParametros = parametros;
    double mejorError = numeric_limits<double>::max();
    vector<string> mejorSubconjunto;
    int mejorNRetrasos = 0;
    //    bool mejorConIndice = false;

    for (const vector<string>& rutasEntradas : subconjuntosRutas) {
        // La combinatoria va a tener un subconjunto vacío
        if (rutasEntradas.empty())
            continue;

        for (int retrasos : retrasosAProbar) {
            //            for (bool conIndice : {true, false}) {
            const Particion particion = cargarTuplas(rutasEntradas,
                                                     rutaDiferencias,
                                                     retrasos,
                                                     6);

            for (int cantidadNeuronas : neuronasPrimerCapa) {
                parametros.estructuraRed = {double(cantidadNeuronas), 6};

                double promedioErrorPromedio = evaluarMLP(parametros,
                                                          particion);

                cout << "El promedio del error cuadrático promedio es: " << promedioErrorPromedio << endl;

                if (promedioErrorPromedio < mejorError) {
                    mejoresParametros = parametros;
                    mejorError = promedioErrorPromedio;
                    mejorSubconjunto = rutasEntradas;
                    mejorNRetrasos = retrasos;
                    //                        mejorConIndice = conIndice;
                }
                //                }
            }
        }
    }

    cout << "El mejor MLP tiene las siguientes características" << endl
         << "Mejor subconjunto: " << endl;
    for (const string& ruta : mejorSubconjunto)
        cout << ruta << endl;
    cout << "Mejor estructura:\n"
         << mejoresParametros.estructuraRed
         << "Mejor error: " << mejorError << endl
         << "Cantidad de retardos en la entrada: " << mejorNRetrasos << endl;
    //         << "Con indice temporal: " << mejorConIndice << endl;

    return 0;
}

double evaluarMLP(const ParametrosMulticapa& parametros,
                  const Particion& particion)
{

    const mat datosEntrenamiento = join_horiz(particion.entrenamiento.tuplasEntrada,
                                              particion.entrenamiento.tuplasSalida);
    vec erroresPromedio(5);

#pragma omp parallel for
    for (int i = 0; i < 5; ++i) {
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
