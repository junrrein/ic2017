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
	//    ventasTuplas = shuffle(ventasTuplas);
    const int nEntrenamiento = ventasTuplas.n_rows * 0.9;
    const mat partEntrenamiento = ventasTuplas.head_rows(nEntrenamiento);
    const mat partPrueba = ventasTuplas.tail_rows(ventasTuplas.n_rows - nEntrenamiento);

    vec estructura = {19, nSalidas};

	vector<mat> pesos;
	double errorEntrenamiento;
	int epocas;
	tie(pesos, errorEntrenamiento, epocas) = ic::entrenarMulticapa(estructura,
                                                                   partEntrenamiento,
                                                                   3000,
                                                                   0.0008,
                                                                   0.2,
                                                                   15,
                                                                   true);

	cout << "El MLP se entrenó en " << epocas << " epocas." << endl
	     << "Error relativo promedio: " << errorEntrenamiento << endl;

    // Pruebas

    mat patrones = partPrueba.head_cols(1 + nEntradas * 3);
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
       << "set multiplot layout 3,2 title 'Predicción usando Ventas + Exportaciones + Importaciones' font ',12'" << endl
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
