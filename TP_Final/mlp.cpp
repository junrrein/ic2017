#include "../guia2/mlp_salida_lineal.cpp"
#include "agrupar_por_tuplas.cpp"
#include "../config.hpp"
#include <gnuplot-iostream.h>

int main()
{
	arma_rng::set_seed_random();

	const string rutaBase = config::sourceDir + "/TP_Final/datos/";
	const string rutaVentas = rutaBase + "Ventas_limpias.csv";
	const string rutaVentasTuplas = rutaBase + "Ventas_limpias_tuplas.csv";

    const int nEntradas = 12;
    const int nSalidas = 1;
	crearTuplas(rutaVentas, nEntradas, nSalidas, rutaVentasTuplas);

	mat ventasTuplas;
    ventasTuplas.load(rutaVentasTuplas);
    ventasTuplas.shed_row(0);
    //    ventasTuplas = shuffle(ventasTuplas);
    mat patrones = ventasTuplas.head_cols(nEntradas);
	mat salidaDeseada = ventasTuplas.tail_cols(nSalidas);

    vec estructura = {7, 1};

	vector<mat> pesos;
	double errorEntrenamiento;
	int epocas;
	tie(pesos, errorEntrenamiento, epocas) = ic::entrenarMulticapa(estructura,
	                                                               ventasTuplas,
                                                                   1000,
                                                                   0.0015,
                                                                   0.2,
	                                                               15);

	cout << "El MLP se entrenó en " << epocas << " epocas." << endl
	     << "Error relativo promedio: " << errorEntrenamiento << endl;

	vec salidaRed(patrones.n_rows);
	for (unsigned int n = 0; n < patrones.n_rows; ++n) {
		salidaRed(n) = as_scalar(ic::salidaMulticapa(pesos,
		                                             patrones.row(n).t())
		                             .back());
	}

	Gnuplot gp;
	gp << "plot " << gp.file1d(salidaDeseada) << " with lines title 'Salida original', "
	   << gp.file1d(salidaRed) << " with lines title 'Salida de la red' lw 2" << endl;

	getchar();

	return 0;
}
