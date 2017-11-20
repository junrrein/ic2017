#include <armadillo>
#include <gnuplot-iostream.h>
#include "../../config.hpp"

using namespace arma;
using namespace std;

int main()
{
    const string rutaBase = config::sourceDir + "/TP_Final/";
    const string rutaArima = rutaBase + "Eleccion_de_modelo/prediccionArima.csv";

    mat datosArima;
    datosArima.load(rutaArima);

    const vec erroresRelativosAbsolutos = abs(datosArima.col(0) - datosArima.col(1))
                                          / datosArima.col(0)
                                          * 100;
    const double promedioErrorPorMes = mean(erroresRelativosAbsolutos);
    const double desvioErrorPorMes = stddev(erroresRelativosAbsolutos);

    const mat errores = join_horiz(vec{promedioErrorPorMes}, vec{desvioErrorPorMes});
    errores.save(rutaBase + "datos/errorArimaSalidaVentas.csv", arma::csv_ascii);

    Gnuplot gp;
    gp << "set xlabel 'Mes (final de la serie)'" << endl
       << "set ylabel 'Patentamientos (unidades)'" << endl
       << "set yrange [0:70000]" << endl
       << "set grid" << endl
       << "set key box opaque bottom center" << endl;

    {
        string errorStr, desvioStr;
        ostringstream ost;
        ost << setprecision(3) << promedioErrorPorMes << " " << desvioErrorPorMes;
        istringstream ist{ost.str()};
        ist >> errorStr >> desvioStr;

        gp << R"(set title "Predicción usando Patentamientos - Modelo ARIMA\n)"
           << R"(EARP = )" << errorStr
           << R"( %, Desvío = )" << desvioStr
           << R"( %")"
           << " font ',11'" << endl
           << "plot " << gp.file1d(datosArima.col(0).eval()) << " with linespoints title 'Salida original', "
           << gp.file1d(datosArima.col(1).eval()) << " with linespoints title 'Salida de la red' lw 2" << endl;
    }

    return 0;
}
