#include "radial_por_lotes.cpp"
#include "../guia1/multicapa.cpp"
#include "../guia1/particionar.cpp"
#include <gnuplot-iostream.h>
#include "../config.hpp"

using namespace std;

int main()
{
    arma_rng::set_seed_random();

    mat datos;
    datos.load(config::sourceDir + "/guia1/icgtp1datos/concentlite.csv");

    ifstream ifs{config::sourceDir + "/guia2/parametrosRbfConcent.txt"};
    ic::ParametrosRBF parametrosRbf;
    if (!(ifs >> parametrosRbf))
        throw runtime_error("Se leyeron mal los parámetros del RBF para el Iris");

    const mat patrones = datos.head_cols(2);
    const mat salidaDeseada = datos.tail_cols(1);

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
    vec errores;
    tie(pesos, errores, ignore, ignore) = ic::entrenarMulticapa(vec{parametrosRbf.estructuraRed(1)},
                                                                datosParaCapaFinal,
                                                                parametrosRbf.nEpocas,
                                                                parametrosRbf.tasaAprendizaje,
                                                                parametrosRbf.inercia,
                                                                parametrosRbf.toleranciaError);

    vector<mat> clases(2);

    for (unsigned int i = 0; i < patrones.n_rows; ++i) {
        if (salidaDeseada(i) == 1)
            clases[0].insert_rows(clases[0].n_rows, patrones.row(i));
        else
            clases[1].insert_rows(clases[1].n_rows, patrones.row(i));
    }

    // Agrupar los centroides en una matriz
    mat matrizCentroides(centroides.size(), 3);
    for (unsigned int i = 0; i < centroides.size(); ++i) {
        matrizCentroides.row(i) = join_horiz(centroides[i], vec{sigmas(i)});
    }

    Gnuplot gp;
    gp << "set title 'Error en esa partición: " << to_string(errores(errores.n_elem - 1)) << "'" << endl
       << "plot " << gp.file1d(clases[0]) << "title 'Clase 1' with points ps 2, "
       << gp.file1d(clases[1]) << "title 'Clase 2' with points ps 2, "
       << gp.file1d(matrizCentroides) << "using 1:2:3 title 'Centroides' with circles" << endl;

    getchar();

    return 0;
}
