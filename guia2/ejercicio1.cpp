#include "radial_por_lotes.cpp"
#include "../guia1/multicapa.cpp"
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

    // 1. Entrenar capa de base radial
    vector<rowvec> centroides;
    vec sigmas;
    tie(centroides, sigmas) = ic::entrenarRadialPorLotes(patrones,
                                                         parametros.estructuraRed(0),
                                                         ic::tipoInicializacion::valoresAlAzar);

    // 2. Calcular la salida de la capa radial para todos los patrones
    mat salidasRadiales(patrones.n_rows, centroides.size());
    for (unsigned int i = 0; i < patrones.n_rows; ++i) {
        salidasRadiales.row(i) = ic::salidaRadial(patrones.row(i), centroides, sigmas);
    }

    datos = join_horiz(salidasRadiales, salidaDeseada);

    // 3. Entrenar la capa de salida de la red usando el algoritmo del multicapa
    vector<mat> pesos;
    tie(pesos, ignore, ignore, ignore) = ic::entrenarMulticapa(vec{parametros.estructuraRed(1)},
                                                               datos,
                                                               parametros.nEpocas,
                                                               parametros.tasaAprendizaje,
                                                               parametros.inercia,
                                                               parametros.toleranciaError);

    // 4. Calcular el error de clasificación de la red final
    datos.load(config::sourceDir + "/guia2/datos/XOR_tst.csv");

    for (unsigned int i = 0; i < patrones.n_rows; ++i) {
        salidasRadiales.row(i) = ic::salidaRadial(patrones.row(i), centroides, sigmas);
    }

    const double errorClasificacion = ic::errorClasificacionMulticapa(pesos, salidasRadiales, salidaDeseada);

    cout << "XOR Capa radial + Capa de salida multicapa" << endl
         << "Tasa de error de clasificación en prueba: " << errorClasificacion << endl;

    // 5. Gráficas
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

    getchar();

    return 0;
}
