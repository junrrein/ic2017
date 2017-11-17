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
	const string rutaExportaciones = rutaBase + "Exportaciones_limpias.csv";
	const string rutaImportaciones = rutaBase + "Importaciones.csv";
	const string rutaTuplas = rutaBase + "Tuplas.csv";

	const int nEntradas = 12;
    const int nSalidas = 6;
	crearTuplas(rutaVentas,
	            rutaExportaciones,
	            rutaImportaciones,
	            nEntradas,
	            nSalidas,
	            rutaTuplas);

	mat ventasTuplas;
	ventasTuplas.load(rutaTuplas);
    ventasTuplas.shed_row(0);
    mat patrones = ventasTuplas.head_cols(1 + nEntradas * 3);
	mat salidaDeseada = ventasTuplas.tail_cols(nSalidas);

    vec estructura = {40, nSalidas};

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
                                                                   1000,
                                                                   0.00025,
                                                                   0.2,
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
        salidaRed(n) = ic::salidaMulticapa(pesos,
                                           salidasRadiales.row(n).t())
                           .back()(0);
    }

    Gnuplot gp;
    gp << "plot " << gp.file1d(salidaDeseada.col(0).eval()) << " with lines title 'Salida original', "
       << gp.file1d(salidaRed) << " with lines title 'Salida de la red' lw 2" << endl;

    getchar();

	return 0;
}
