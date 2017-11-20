#include <armadillo>
#include <gnuplot-iostream.h>
#include "../../config.hpp"

using namespace arma;
using namespace std;

int main()
{
    const string rutaBase = config::sourceDir + "/TP_Final/";
    const string rutaArima = rutaBase + "Eleccion_de_modelo/prediccionArimaDifRel.csv";

    mat datosArima;
    datosArima.load(rutaArima);
    vec salidaDeseada = datosArima.col(0);
    vec salidaModelo = datosArima.col(1);

    // Pasar datos de diferencias relativas a patentamientos en unidades
    const vec ventas = {
        49023,
        44305,
        37316,
        23008,
        42900,
        36788,
        43908,
        45160,
        44184,
        39445,
        48209,
        54326,
        52254,
        45313,
    };

    salidaModelo = salidaModelo / 100 % ventas + ventas;
    salidaDeseada = salidaDeseada / 100 % ventas + ventas;

    // Calculo de error
    const vec erroresRelativosAbsolutos = abs(salidaDeseada - salidaModelo)
                                          / salidaDeseada
                                          * 100;
    const double promedioErrorPorMes = mean(erroresRelativosAbsolutos);
    const double desvioErrorPorMes = stddev(erroresRelativosAbsolutos);

    const mat errores = join_horiz(vec{promedioErrorPorMes}, vec{desvioErrorPorMes});
    errores.save(rutaBase + "datos/errorArimaSalidaDifRel.csv", arma::csv_ascii);

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
           << "plot " << gp.file1d(salidaDeseada) << " with linespoints title 'Salida original', "
           << gp.file1d(salidaModelo) << " with linespoints title 'Salida de la red' lw 2" << endl;
    }

    getchar();

    return 0;
}
