#include "../guia2/radial_por_lotes.cpp"
#include "../guia2/mlp_salida_lineal.cpp"
#include "agrupar_por_tuplas.cpp"
#include "../config.hpp"
#include <gnuplot-iostream.h>

int main()
{
    arma_rng::set_seed_random();

    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    const string rutaVentas = rutaBase + "Ventas_limpias.csv";
    const string rutaVentasTuplas = rutaBase + "Ventas_tuplas.csv";

    crearTuplas(rutaVentas, 24, 1, rutaVentasTuplas);

    mat ventasTuplas;
    ventasTuplas.load(rutaVentasTuplas);
    mat patrones = ventasTuplas.head_cols(24);
    mat salidaDeseada = ventasTuplas.tail_cols(1);

    vec estructura = {10, 1};

    vector<rowvec> centroides;
    vec sigmas;
    tie(centroides, sigmas) = ic::entrenarRadialPorLotes(patrones,
                                                         estructura(0),
                                                         ic::tipoInicializacion::patronesAlAzar);

    mat salidasRadiales(patrones.n_rows, centroides.size());
    for (unsigned int j = 0; j < patrones.n_rows; ++j)
        salidasRadiales.row(j) = ic::salidaRadial(patrones.row(j), centroides, sigmas);

    mat datosParaCapaFinal = join_horiz(salidasRadiales, salidaDeseada);

    vector<mat> pesos;
    double errorEntrenamiento;
    int epocas;
    tie(pesos, errorEntrenamiento, epocas) = ic::entrenarMulticapa(vec{estructura(1)},
                                                                   datosParaCapaFinal,
                                                                   2000,
                                                                   0.1,
                                                                   0.3,
                                                                   15);

    cout << "El MLP se entrenó en " << epocas << " epocas." << endl
         << "Error relativo promedio: " << errorEntrenamiento << endl;

    salidasRadiales = mat(patrones.n_rows, centroides.size());
    for (unsigned int j = 0; j < patrones.n_rows; ++j)
        salidasRadiales.row(j) = ic::salidaRadial(patrones.row(j),
                                                  centroides,
                                                  sigmas);

    vec salidaRed(patrones.n_rows);
    for (unsigned int n = 0; n < patrones.n_rows; ++n) {
        salidaRed(n) = as_scalar(ic::salidaMulticapa(pesos,
                                                     salidasRadiales.row(n).t())
                                     .back());
    }

    Gnuplot gp;
    gp << "plot " << gp.file1d(salidaDeseada) << " with lines title 'Salida original', "
       << gp.file1d(salidaRed) << " with lines title 'Salida de la red'" << endl;

    cout << "La media de la salida original es: " << mean(salidaDeseada) << endl;

    getchar();

    return 0;
}
