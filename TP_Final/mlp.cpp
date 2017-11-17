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
    const int nSalidas = 6;
	crearTuplas(rutaVentas, nEntradas, nSalidas, rutaVentasTuplas);

    mat ventasTuplas;
    ventasTuplas.load(rutaVentasTuplas);
    ventasTuplas.shed_row(0);
    //    ventasTuplas = shuffle(ventasTuplas);
    const int nEntrenamiento = ventasTuplas.n_rows * 0.9;
    const mat partEntrenamiento = ventasTuplas.head_rows(nEntrenamiento);
    const mat partPrueba = ventasTuplas.tail_rows(ventasTuplas.n_rows - nEntrenamiento);

    vec estructura = {17, nSalidas};

	vector<mat> pesos;
	double errorEntrenamiento;
	int epocas;
	tie(pesos, errorEntrenamiento, epocas) = ic::entrenarMulticapa(estructura,
                                                                   partEntrenamiento,
                                                                   2000,
                                                                   0.00075,
                                                                   0.25,
                                                                   15,
                                                                   false);

	cout << "El MLP se entrenó en " << epocas << " epocas." << endl
	     << "Error relativo promedio: " << errorEntrenamiento << endl;

    // Pruebas

    mat patrones = partPrueba.head_cols(1 + nEntradas);
    mat salidaDeseada = partPrueba.tail_cols(nSalidas);
    vector<vec> salidaRed(nSalidas, vec(patrones.n_rows));

    for (unsigned int n = 0; n < patrones.n_rows; ++n) {
        vec salidas = ic::salidaMulticapa(pesos,
                                          patrones.row(n).t())
                          .back();

        for (int j = 0; j < nSalidas; ++j)
            salidaRed.at(j)(n) = salidas(j);
    }

    Gnuplot gp;
    gp << "set terminal qt size 750,700" << endl
       << "set multiplot layout 3,2 title 'Predicción usando Ventas' font ',12'" << endl
       << "set xlabel 'Mes'" << endl
       << "set ylabel 'Ventas normalizadas'" << endl
       << "set yrange [0:1]" << endl
       << "set grid" << endl;

    for (unsigned int i = 0; i < salidaRed.size(); ++i) {
        gp << "set title 'Predicción - " << i + 1 << " meses hacia adelante'" << endl
           << "plot " << gp.file1d(salidaDeseada.col(i).eval()) << " with linespoints title 'Salida original', "
           << gp.file1d(salidaRed.at(i)) << " with linespoints title 'Salida de la red' lw 2" << endl;
    }

    getchar();

	return 0;
}
