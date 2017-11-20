#include <armadillo>
#include <gnuplot-iostream.h>
#include <../../config.hpp>

using namespace arma;
using namespace std;

int main()
{
    // Carga de datos
    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    mat errorMlpVentas, errorMlpDiferencias, errorRbfVentas, errorArimaVentas, errorArimaDifRel;
    errorMlpVentas.load(rutaBase + "errorMlpSalidaVentas.csv");
    errorMlpDiferencias.load(rutaBase + "errorMlpSalidaDifRel.csv");
    errorRbfVentas.load(rutaBase + "errorRbfSalidaVentas.csv");
    errorArimaVentas.load(rutaBase + "errorArimaSalidaVentas.csv");
    errorArimaDifRel.load(rutaBase + "errorArimaSalidaDifRel.csv");

    // Agregar indice mensual
    const vec indices = linspace(1, 6, 6);
    errorMlpVentas = join_horiz(indices - 0.1, errorMlpVentas);
    errorMlpVentas(0, 0) = 0.8;
    errorMlpDiferencias = join_horiz(indices, errorMlpDiferencias);
    errorMlpDiferencias(0, 0) = 0.9;
    errorRbfVentas = join_horiz(indices + 0.1, errorRbfVentas);
    errorRbfVentas(0, 0) = 1;
    errorArimaVentas = join_horiz(vec{1.1}, errorArimaVentas);
    errorArimaDifRel = join_horiz(vec{1.2}, errorArimaDifRel);

    Gnuplot gp;
    gp << "set title 'Comparación del desempeño de distintos modelos de predicción' font ',12'" << endl
       << "set xlabel 'Plazo de predicción (meses hacia adelante)' font ',11'" << endl
       << "set ylabel 'EARP (%)' font ',11'" << endl
       << "set xrange [0:7]" << endl
       << "set grid ytics" << endl
       << "set key box opaque width 3" << endl
       << "plot "
       << gp.file1d(errorMlpVentas) << " with errorbars title 'MLP 1' lw 2, "
       << gp.file1d(errorMlpDiferencias) << " with errorbars title 'MLP 2' lw 2, "
       << gp.file1d(errorRbfVentas) << " with errorbars title 'RBF' lw 2, "
       << gp.file1d(errorArimaVentas) << " with errorbars title 'ARIMA 1' lw 2, "
       << gp.file1d(errorArimaDifRel) << " with errorbars title 'ARIMA 2' lw 2" << endl;

    getchar();

    return 0;
}
