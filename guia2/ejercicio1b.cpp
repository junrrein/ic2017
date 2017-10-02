#include "radial_por_lotes.cpp"
#include "../guia1/multicapa.cpp"
#include "../guia1/particionar.cpp"
#include <gnuplot-iostream.h>
#include "../config.hpp"

using namespace std;

int main()
{
    arma_rng::set_seed_random();

    const string rutaParticiones = config::sourceDir + "/guia1/icgtp1datos/particionesIrisKOut/";
    const vector<ic::Particion> particiones = ic::cargarParticiones(rutaParticiones, 10);
    mat datos;
    datos.load(config::sourceDir + "/guia2/datos/irisbin.csv");

    // Entrenamiento con MLP
    ifstream ifs{config::sourceDir + "/guia2/parametrosMlpIris.txt"};
    ic::ParametrosMulticapa parametrosMlp;
    if (!(ifs >> parametrosMlp))
        throw runtime_error("Se cargaron mal los parámetros del MLP para el Iris");

    // Estructura de datos para guardar los pesos de todas las redes
    // (una para cada partición)
    using Pesos = vector<mat>;
    vector<Pesos> pesosRedesMlp;
    pesosRedesMlp.resize(particiones.size());

    // Vamos a tomar el tiempo acumulado que se tarda en entrenar todas las particiones
    wall_clock reloj;
    reloj.tic();

    for (unsigned int i = 0; i < particiones.size(); ++i) {
        tie(pesosRedesMlp[i], ignore, ignore, ignore) = ic::entrenarMulticapa(parametrosMlp.estructuraRed,
                                                                              datos.rows(particiones[i].first),
                                                                              parametrosMlp.nEpocas,
                                                                              parametrosMlp.tasaAprendizaje,
                                                                              parametrosMlp.inercia,
                                                                              parametrosMlp.toleranciaError);
    }

    const double tiempoMlp = reloj.toc();

    // Cálculo del error del MLP
    vec erroresMlp;
    erroresMlp.set_size(particiones.size());

    for (unsigned int i = 0; i < particiones.size(); ++i)
        erroresMlp(i) = ic::errorClasificacionMulticapa(pesosRedesMlp[i],
                                                        datos.rows(particiones[i].second));

    // Entrenamiento con RBF
    ifs.close();
    ifs.open(config::sourceDir + "/guia2/parametrosRbfIris.txt");
    ic::ParametrosRBF parametrosRbf;
    if (!(ifs >> parametrosRbf))
        throw runtime_error("Se leyeron mal los parámetros del RBF para el Iris");

    // Estructura de datos para guardar los pesos de todas las redes
    // (una para cada partición)
    using Centroides = vector<rowvec>;
    vector<Centroides> centroidesRedesRbf;
    vector<vec> sigmasRedesRbf;
    vector<Pesos> pesosCapasFinalesRbf;
    centroidesRedesRbf.resize(particiones.size());
    sigmasRedesRbf.resize(particiones.size());
    pesosCapasFinalesRbf.resize(particiones.size());

    // Vamos a tomar el tiempo acumulado que se tarda en entrenar todas las particiones
    reloj.tic();

    for (unsigned int i = 0; i < particiones.size(); ++i) {
        const mat datosParticion = datos.rows(particiones[i].first);
        const mat patrones = datosParticion.head_cols(4);
        const mat salidaDeseada = datosParticion.tail_cols(3);

        tie(centroidesRedesRbf[i], sigmasRedesRbf[i]) = ic::entrenarRadialPorLotes(patrones,
                                                                                   parametrosRbf.estructuraRed(0),
                                                                                   ic::tipoInicializacion::conjuntosAleatorios);

        mat salidasRadiales(patrones.n_rows, centroidesRedesRbf[i].size());
        for (unsigned int j = 0; j < patrones.n_rows; ++j)
            salidasRadiales.row(j) = ic::salidaRadial(patrones.row(j), centroidesRedesRbf[i], sigmasRedesRbf[i]);

        const mat datosParaCapaFinal = join_horiz(salidasRadiales, salidaDeseada);

        tie(pesosCapasFinalesRbf[i], ignore, ignore, ignore) = ic::entrenarMulticapa(vec{parametrosRbf.estructuraRed(1)},
                                                                                     datosParaCapaFinal,
                                                                                     parametrosRbf.nEpocas,
                                                                                     parametrosRbf.tasaAprendizaje,
                                                                                     parametrosRbf.inercia,
                                                                                     parametrosRbf.toleranciaError);
    }

    const double tiempoRbf = reloj.toc();

    // Cálculo del error del RBF
    vec erroresRbf;
    erroresRbf.set_size(particiones.size());

    for (unsigned int i = 0; i < particiones.size(); ++i) {
        const mat datosParticion = datos.rows(particiones[i].second);
        const mat patrones = datosParticion.head_cols(4);
        const mat salidaDeseada = datosParticion.tail_cols(3);

        mat salidasRadiales(patrones.n_rows, centroidesRedesRbf[i].size());
        for (unsigned int j = 0; j < patrones.n_rows; ++j)
            salidasRadiales.row(j) = ic::salidaRadial(patrones.row(j),
                                                      centroidesRedesRbf[i],
                                                      sigmasRedesRbf[i]);

        const mat datosParaCapaFinal = join_horiz(salidasRadiales, salidaDeseada);

        erroresRbf(i) = ic::errorClasificacionMulticapa(pesosCapasFinalesRbf[i],
                                                        salidasRadiales,
                                                        salidaDeseada);
    }

    cout << "Iris MLP con inercia " << parametrosMlp.inercia << " (19 parámetros)" << endl
         << "Tiempo que tarda en entrenar 10 particiones: " << tiempoMlp << " segundos." << endl
         << "Error de clasificación promedio en pruebas: " << mean(erroresMlp) << endl
         << "Desvío de lo anterior: " << stddev(erroresMlp) << endl
         << "\nIris RBF (19 parámetros)" << endl
         << "Tiempo que tarda en entrenar 10 particiones: " << tiempoRbf << " segundos." << endl
         << "Error de clasificación promedio en pruebas: " << mean(erroresRbf) << endl
         << "Desvío de lo anterior: " << stddev(erroresRbf) << endl;

    // Graficar patrones y centroides proyectados en R^2
    // Separar patrones en clases
    const int particion = 1;
    const mat datosPrueba = datos.rows(particiones[particion].first);
    const mat patrones = datosPrueba.head_cols(2);
    const mat salidaDeseada = datosPrueba.tail_cols(3);
    vector<mat> clases(3);

    for (unsigned int i = 0; i < patrones.n_rows; ++i) {
        const int indiceMayor = index_max(salidaDeseada.row(i));

        clases[indiceMayor].insert_rows(clases[indiceMayor].n_rows, patrones.row(i));
    }

    // Agrupar los centroides en una matriz
    const vector<rowvec> centroides = centroidesRedesRbf[particion];
    mat matrizCentroides(centroides.size(), 3);
    for (unsigned int i = 0; i < centroides.size(); ++i) {
        matrizCentroides.row(i) = join_horiz(centroides[i].head(2), vec{sigmasRedesRbf[particion](i)});
    }

    Gnuplot gp;
    gp << "set title 'Error en esa partición: " << erroresRbf(particion) << "'" << endl
       << "plot " << gp.file1d(clases[0]) << "title 'Clase 1' with points ps 2, "
       << gp.file1d(clases[1]) << "title 'Clase 2' with points ps 2, "
       << gp.file1d(clases[2]) << "title 'Clase 3' with points ps 2, "
       << gp.file1d(matrizCentroides) << "using 1:2:(2*($3)) title 'Centroides' with circles" << endl;

    getchar();

    // Graficar evolucion del error durante el entrenamiento del RBF para una partición
    /*
    vec erroresClasificacion;
    vec erroresCuadraticos;
    int epocas;

    {
        const mat datosParticion = datos.rows(particiones[0].first);
        const mat patrones = datosParticion.head_cols(4);
        const mat salidaDeseada = datosParticion.tail_cols(3);

        vector<rowvec> centroides;
        vec sigmas;
        tie(centroides, sigmas) = ic::entrenarRadialPorLotes(patrones,
                                                             parametrosRbf.estructuraRed(0),
                                                             ic::tipoInicializacion::conjuntosAleatorios);

        mat salidasRadiales(patrones.n_rows, centroides.size());
        for (unsigned int j = 0; j < patrones.n_rows; ++j)
            salidasRadiales.row(j) = ic::salidaRadial(patrones.row(j), centroides, sigmas);

        const mat datosParaCapaFinal = join_horiz(salidasRadiales, salidaDeseada);

        vector<mat> pesos;
        tie(pesos,
            erroresClasificacion,
            erroresCuadraticos,
            epocas)
            = ic::entrenarMulticapa(vec{parametrosRbf.estructuraRed(1)},
                                    datosParaCapaFinal,
                                    parametrosRbf.nEpocas,
                                    parametrosRbf.tasaAprendizaje,
                                    parametrosRbf.inercia,
                                    parametrosRbf.toleranciaError);
    }

    Gnuplot gp;
    gp << "set multiplot layout 2, 1 title 'Entrenamiento de la capa final del RBF' font ',14'" << endl
       << "set xrange [1:" << epocas + 1 << "]" << endl
       << "set yrange [0:*]" << endl
       << "unset key" << endl
       << "set grid linewidth 1" << endl
       << "set title 'Evolución del error de clasificación'" << endl
       << "set xlabel 'Epoca'" << endl
       << "set ylabel 'Tasa de error (%)'" << endl
       << "plot " << gp.file1d(erroresClasificacion) << "with lines" << endl
       << "set title 'Evolución de la suma del error cuadrático'" << endl
       << "set ylabel 'Suma del error cuadrático'" << endl
       << "plot " << gp.file1d(erroresCuadraticos) << "with lines" << endl;

    getchar();
*/

    return 0;
}
