#include "construir_tuplas.cpp"
#include "../../guia2/radial_por_lotes.cpp"
#include "../../guia2/mlp_salida_lineal.cpp"
#include "../../config.hpp"

using namespace ic;

// Evalúa una cierta configuración de un MLP 10 veces.
// Devuelve el promedio del error cuadrático promedio
// en una partición de evaluación.
// De los datos que se reciben, se descarta el último 10%
// (se dejan para pruebas posteriores), y del 90% restante
// se aparta el 20% para hacer la evaluación.
double evaluarRBF(const ParametrosRBF& parametros,
                  double sigma,
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
    vector<int> neuronasPrimerCapa(18);
    iota(neuronasPrimerCapa.begin(), neuronasPrimerCapa.end(), 7);
    vector<unsigned int> retrasosAProbar(12);
    iota(retrasosAProbar.begin(), retrasosAProbar.end(), 4);

    ParametrosRBF parametros;
    parametros.nEpocas = 2000;
    parametros.tasaAprendizaje = 0.00075;
    parametros.inercia = 0.2;
    parametros.toleranciaError = 10;
    ParametrosRBF mejoresParametros = parametros;
    double mejorError = numeric_limits<double>::max();
    vector<string> mejorSubconjunto;
    int mejorNRetrasos = 0;
    double mejorSigma = 0;
    bool mejorConIndice = false;

    for (const vector<string>& rutasEntradas : subconjuntosRutas) {
        // La combinatoria va a tener un subconjunto vacío
        if (rutasEntradas.empty())
            continue;

        for (int retrasos : retrasosAProbar) {
            for (bool conIndice : {true, false}) {
                const Particion particion = cargarTuplas(rutasEntradas,
                                                         rutaVentas,
                                                         retrasos,
                                                         6,
                                                         conIndice);

                for (int cantidadNeuronas : neuronasPrimerCapa) {
                    for (double sigma : {0.1, 0.2, 0.3, 0.4, 0.5}) {
                        parametros.estructuraRed = {double(cantidadNeuronas), 6};

                        double promedioErrorPromedio = evaluarRBF(parametros,
                                                                  sigma,
                                                                  particion);

                        cout << "El promedio del error cuadrático promedio es: " << promedioErrorPromedio << endl;

                        if (promedioErrorPromedio < mejorError) {
                            mejoresParametros = parametros;
                            mejorError = promedioErrorPromedio;
                            mejorSubconjunto = rutasEntradas;
                            mejorNRetrasos = retrasos;
                            mejorSigma = sigma;
                            mejorConIndice = conIndice;
                        }
                    }
                }
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
         << "Cantidad de retardos en la entrada: " << mejorNRetrasos << endl
         << "Mejor sigma: " << mejorSigma << endl
         << "Con indice temporal: " << mejorConIndice << endl;

    return 0;
}

double evaluarRBF(const ParametrosRBF& parametros,
                  double sigma,
                  const Particion& particion)
{
    vec erroresPromedio(5);

#pragma omp parallel for
    for (int i = 0; i < 5; ++i) {
        // Entrenamiento
        vector<rowvec> centroides;
        vec sigmas;
        tie(centroides, sigmas)
            = entrenarRadialPorLotes(particion.entrenamiento.tuplasEntrada,
                                     parametros.estructuraRed(0),
                                     tipoInicializacion::patronesAlAzar,
                                     sigma);

        mat salidasRadiales(particion.entrenamiento.tuplasEntrada.n_rows,
                            centroides.size());
        for (unsigned int j = 0; j < particion.entrenamiento.tuplasEntrada.n_rows; ++j)
            salidasRadiales.row(j) = salidaRadial(particion.entrenamiento.tuplasEntrada.row(j),
                                                  centroides,
                                                  sigmas);

        mat datosCapaFinal = join_horiz(salidasRadiales,
                                        particion.entrenamiento.tuplasSalida);

        vector<mat> pesos;
        tie(pesos, ignore, ignore) = entrenarMulticapa(vec{parametros.estructuraRed(1)},
                                                       datosCapaFinal,
                                                       parametros.nEpocas,
                                                       parametros.tasaAprendizaje,
                                                       parametros.inercia,
                                                       parametros.toleranciaError,
                                                       true);

        // Prueba
        salidasRadiales = mat(particion.evaluacion.tuplasEntrada.n_rows,
                              centroides.size());
        for (unsigned int j = 0; j < particion.evaluacion.tuplasEntrada.n_rows; ++j)
            salidasRadiales.row(j) = salidaRadial(particion.evaluacion.tuplasEntrada.row(j),
                                                  centroides,
                                                  sigmas);

        double errorTotal = errorCuadraticoMulticapa(pesos,
                                                     salidasRadiales,
                                                     particion.evaluacion.tuplasSalida);
        double errorPromedio = errorTotal / particion.evaluacion.tuplasEntrada.n_rows;
        erroresPromedio(i) = errorPromedio;
    }

    return mean(erroresPromedio);
}
