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
    vec salidaRed(patrones.n_rows);
	for (unsigned int n = 0; n < patrones.n_rows; ++n) {
        salidaRed(n) = ic::salidaMulticapa(pesos,
                                           patrones.row(n).t())
                           .back()(0);
	}

	Gnuplot gp;
    gp << "plot " << gp.file1d(salidaDeseada.col(0).eval()) << " with lines title 'Salida original', "
	   << gp.file1d(salidaRed) << " with lines title 'Salida de la red' lw 2" << endl;

	getchar();

	return 0;
}
