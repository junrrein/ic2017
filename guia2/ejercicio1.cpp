#include "radial_por_lotes.cpp"
#include <gnuplot-iostream.h>
#include "../config.hpp"

int main()
{
    arma_rng::set_seed_random();

    mat datos;
    datos.load(config::sourceDir + "/guia2/datos/XOR_trn.csv");

    ic::ParametrosRBF parametros;
    ifstream ifs{config::sourceDir + "/guia2/parametrosRbfXor.txt"};
    if (!ifs)
        throw runtime_error("No se pudo abrir el archivo");

    if (!(ifs >> parametros))
        throw runtime_error("Se leyeron mal los parametros del RBF para el XOR");

    const mat patrones = datos.head_cols(2);
    const vec salidaDeseada = datos.tail_cols(1);
    vector<rowvec> centroides;
    vec sigmas;
    tie(centroides, sigmas) = ic::entrenarRadialPorLotes(patrones,
                                                         parametros.estructuraRed(0),
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
       << "plot " << gp.file1d(verdaderos) << "title 'Verdaderos' with points lt rgb 'cyan', "
       << gp.file1d(falsos) << "title 'Falsos' with points lt rgb 'green', "
       << gp.file1d(matrizCentroides) << "title 'Centroides' with points pt 6 ps 2 lw 3 lt rgb 'black'" << endl;

    return 0;
}
