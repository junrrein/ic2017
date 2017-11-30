#include <armadillo>
#include <gnuplot-iostream.h>
#include <../../config.hpp>

using namespace arma;
using namespace std;

int main()
{
    // Carga de datos
    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    mat errorMlpVentas, errorMlpDiferencias, errorRbfVentas, errorRbfDiferencias, errorArimaVentas, errorArimaDiferencias;
    errorMlpVentas.load(rutaBase + "errorMlpSalidaVentas.csv");
    errorMlpDiferencias.load(rutaBase + "errorMlpSalidaDifRel.csv");
    errorRbfVentas.load(rutaBase + "errorRbfSalidaVentas.csv");
    errorRbfDiferencias.load(rutaBase + "errorRbfSalidaDifRel.csv");
    //    errorArimaVentas.load(rutaBase + "errorArimaSalidaVentas.csv");
    //    errorArimaDiferencias.load(rutaBase + "errorArimaSalidaDifRel.csv");

    // Agregar indice mensual
    const vec indices = linspace(1, errorMlpVentas.n_rows, errorMlpVentas.n_rows);
    errorMlpVentas = join_horiz(indices - 0.15, errorMlpVentas);
    errorMlpDiferencias = join_horiz(indices - 0.05, errorMlpDiferencias);
    errorRbfVentas = join_horiz(indices + 0.05, errorRbfVentas);
    errorRbfDiferencias = join_horiz(indices + 0.15, errorRbfDiferencias);
    //    errorArimaVentas = join_horiz(indices + 0.15, errorArimaVentas);
    //    errorArimaDiferencias = join_horiz(indices + 0.25, errorArimaDiferencias);

    Gnuplot gp;
    gp << "set title 'Comparación del desempeño de distintos modelos de predicción' font ',12'" << endl
       << "set xlabel 'Plazo de predicción (meses hacia adelante)' font ',11'" << endl
       << "set ylabel 'EARP (%)' font ',11'" << endl
       << "set xrange [0:4]" << endl
       << "set yrange [0:50]" << endl
       << "set grid ytics" << endl
       << "set key box opaque width 3" << endl
       << "plot "
       << gp.file1d(errorMlpVentas) << " with errorbars title 'MLP 1' lw 2, "
       << gp.file1d(errorMlpDiferencias) << " with errorbars title 'MLP 2' lw 2, "
       << gp.file1d(errorRbfVentas) << " with errorbars title 'RBF 1' lw 2, "
       << gp.file1d(errorRbfDiferencias) << " with errorbars title 'RBF 2' lw 2" << endl;

    getchar();

    return 0;
}
