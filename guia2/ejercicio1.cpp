#include "radial_por_lotes.cpp"
#include <gnuplot-iostream.h>
#include "../config.hpp"

int main()
{
    arma_rng::set_seed_random();

    mat datos;
    datos.load(config::sourceDir + "/guia2/datos/XOR_trn.csv");

    const mat patrones = datos.head_cols(2);
    const vec salidaDeseada = datos.tail_cols(1);
    vector<rowvec> centroides;
    vec sigmas;
    tie(centroides, sigmas) = ic::entrenarRadialPorLotes(patrones,
                                                         6,
                                                         ic::tipoInicializacion::valoresAlAzar);

    mat matrizCentroides;
    for (const rowvec& centroide : centroides) {
        matrizCentroides.insert_rows(matrizCentroides.n_rows, centroide);
    }

    const mat verdaderos = patrones.rows(find(salidaDeseada == 1));
    const mat falsos = patrones.rows(find(salidaDeseada == -1));

    Gnuplot gp;
    gp << "set title 'XOR base radial' font ',13'" << endl
       << "set xlabel 'x_1'" << endl
       << "set ylabel 'x_2'" << endl
       << "set grid" << endl
       << "plot " << gp.file1d(verdaderos) << "title 'Verdaderos' with points lt rgb 'blue', "
       << gp.file1d(falsos) << "title 'Falsos' with points lt rgb 'red', "
       << gp.file1d(matrizCentroides) << "title 'Centroides' with points pt 5 ps 2 lt rgb 'green'" << endl;

    return 0;
}
