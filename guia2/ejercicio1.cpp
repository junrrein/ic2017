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

    mat patrones = datos.head_cols(2);
    vec salidaDeseada = datos.tail_cols(1);

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
    patrones = datos.head_cols(2);
    salidaDeseada = datos.tail_cols(1);
    salidasRadiales.clear();
    salidasRadiales.set_size(patrones.n_rows, centroides.size());

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

    // Se calcula la salida de la red híbrida para todo el plano,
    // para luego graficar la frontera de decisión.
    const int N = 125;
    mat puntosSuperficie(N * N, 3);
    const vec x = linspace(-1.5, 1.5, N);
    const vec y = linspace(-1.5, 1.5, N);

    for (unsigned int i = 0; i < x.n_elem; ++i) {
        for (unsigned int j = 0; j < y.n_elem; ++j) {
            const rowvec salidaRBF = ic::salidaRadial(rowvec{x(i), y(j)}, centroides, sigmas);
            const vector<vec> salidasRed = ic::salidaMulticapa(pesos, salidaRBF.t());
            const double salidaFinal = salidasRed.back()(0);

            puntosSuperficie.row(N * i + j) = rowvec{x(i), y(j), salidaFinal};
        }
    }

    Gnuplot gp;
    gp << "set title 'XOR base radial' font ',13'" << endl
       << "set xlabel 'x_1'" << endl
       << "set ylabel 'x_2'" << endl
       << "set key box opaque outside width 3" << endl
       << "set grid" << endl

       // Se plotea la frontera de decisión a un archivo
       << "set contour base" << endl
       << "set cntrparam levels discrete 0" << endl
       << "set dgrid3d " << N << "," << N << endl
       << "unset surface" << endl
       << "set table 'fronteraRbfXor.dat'" << endl
       << "splot " << gp.file1d(puntosSuperficie) << endl
       << "unset table" << endl
       << "set surface" << endl
       << "unset dgrid3d" << endl
       << "unset contour" << endl

       << "plot " << gp.file1d(verdaderos) << "title 'Verdaderos' with points lt rgb 'cyan', "
       << gp.file1d(falsos) << "title 'Falsos' with points lt rgb 'green', "
       << gp.file1d(matrizCentroides) << "title 'Centroides' with points pt 2 ps 1 lw 2 lt rgb 'black', "
       << "'fronteraRbfXor.dat' title 'Frontera de decisión' with lines lw 2.5 lt rgb 'magenta'" << endl;

    getchar();

    return 0;
}
