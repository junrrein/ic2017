#include <armadillo>
#include <gnuplot-iostream.h>
#include <../../config.hpp>

using namespace arma;
using namespace std;

int main()
{
    // Carga de datos
    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    mat errorMlpVentas, errorMlpDiferencias;
    errorMlpVentas.load(rutaBase + "/errorMlpSalidaVentas.csv");
    errorMlpDiferencias.load(rutaBase + "/errorMlpSalidaDifRel.csv");

    // Agregar indice mensual
    const vec indices = linspace(1, 6, 6);
    errorMlpVentas = join_horiz(indices - 0.05, errorMlpVentas);
    errorMlpDiferencias = join_horiz(indices + 0.05, errorMlpDiferencias);

    Gnuplot gp;
    gp << "set title 'Comparación del desempeño de distintos modelos de predicción' font ',12'" << endl
       << "set xlabel 'Meses hacia adelante' font ',11'" << endl
       << "set ylabel 'EARP (%)' font ',11'" << endl
       << "set xrange [0:7]" << endl
       << "set grid ytics" << endl
       << "set key box opaque" << endl
       << "plot "
       << gp.file1d(errorMlpVentas) << " with errorbars title 'Modelo 1' lw 2, "
       << gp.file1d(errorMlpDiferencias) << " with errorbars title 'Modelo 2' lw 2" << endl;

    getchar();

    return 0;
}
