#include <armadillo>
#include <gnuplot-iostream.h>
#include "../../config.hpp"

using namespace arma;
using namespace std;

int main()
{
    const string rutaBase = config::sourceDir + "/TP_Final/";

    const vec diferencias = {-9.62, -15.77, -38.34, 86.46, -14.25, 19.35, 2.85, -2.16, -10.73, 22.22, 12.69, -3.81, -13.28, -7.45, -30.71, 135.69, -38.21, 34.82, -15.64};
    const vec ventas = {44305, 37316, 23008, 42900, 36788, 43908, 45160, 44184, 39445, 48209, 54326, 52254, 45313, 41938, 29057, 68485, 42320, 57054, 48130};
    const vec ventasAnteriores = {49023, 44305, 37316, 23008, 42900, 36788, 43908, 45160, 44184, 39445, 48209, 54326, 52254, 45313};
    mat salidasArima = {{-14.90, -24.36, -23.95, 125.77, -33.78, 13.93},
                        {-29.77, -26.89, 121.01, -37.46, 10.07, 3.56},
                        {-26.12, 125.93, -34.78, 13.58, 6.62, -7.65},
                        {129.74, -35.93, 15.10, 6.81, -7.18, 18.21},
                        {-25.35, 11.50, 10.78, -7.46, 18.88, 7.36},
                        {7.63, 11.02, -8.29, 18.98, 7.34, -7.40},
                        {8.95, -7.22, 18.18, 7.41, -7.54, 19.45},
                        {-7.53, 16.40, 6.45, -8.77, 18.28, -9.39},
                        {16.72, 7.94, -8.08, 19.39, -8.63, -14.37},
                        {14.68, -10.22, 21.73, -8.55, -13.61, -35.27},
                        {-11.99, 22.57, -9.09, -13.42, -35.30, 81.68},
                        {16.45, -7.11, -15.49, -35.23, 81.30, -12.59},
                        {-2.01, -16.93, -33.53, 81.11, -12.25, 18.75},
                        {-14.08, -34.34, 82.07, -12.30, 18.93, 3.44}};

    salidasArima.col(0) = salidasArima.col(0) / 100 % ventasAnteriores + ventasAnteriores;
    for (unsigned int i = 1; i < salidasArima.n_cols; ++i)
        salidasArima.col(i) = salidasArima.col(i) / 100 % salidasArima.col(i - 1) + salidasArima.col(i - 1);

    // Calculo de error
    vec promedioErrorPorMes(6);
    vec desvioErrorPorMes(6);
    for (int i = 0; i < 6; ++i) {
        const vec pedazoVentas = ventas(span(i, i + salidasArima.n_rows - 1));
        const vec erroresRelativosAbsolutos = (abs(salidasArima.col(i) - pedazoVentas))
                                              / pedazoVentas
                                              * 100;
        promedioErrorPorMes(i) = mean(erroresRelativosAbsolutos);
        desvioErrorPorMes(i) = stddev(erroresRelativosAbsolutos);
    }

    const mat errores = join_horiz(vec{promedioErrorPorMes}, vec{desvioErrorPorMes});
    errores.save(rutaBase + "datos/errorArimaSalidaDifRel.csv", arma::csv_ascii);

    Gnuplot gp;
    gp << "set terminal qt size 1200,600" << endl
       << "set multiplot layout 2,3 title 'Predicción usando Diferencias relativas de patentamientos - ARIMA' font ',12'" << endl
       << "set xlabel 'Mes (final de la serie)'" << endl
       << "set ylabel 'Patentamientos (unidades)'" << endl
       << "set yrange [0:70000]" << endl
       << "set grid" << endl
       << "set key box opaque bottom center" << endl;

    {
        string errorStr, desvioStr;
        ostringstream ost;
        ost << setprecision(3) << promedioErrorPorMes(0) << " " << desvioErrorPorMes(0);
        istringstream ist{ost.str()};
        ist >> errorStr >> desvioStr;

        gp << R"(set title "1 mes hacia adelante\n)"
           << R"(EARP = )" << errorStr
           << R"( %, Desvío = )" << desvioStr
           << R"( %")"
           << " font ',11'" << endl
           << "plot " << gp.file1d(ventas(span(0, salidasArima.n_rows - 1)).eval()) << " with linespoints title 'Salida original', "
           << gp.file1d(salidasArima.col(0).eval()) << " with linespoints title 'Salida de la red' lw 2" << endl;
    }

    for (unsigned int i = 1; i < salidasArima.n_cols; ++i) {
        string errorStr, desvioStr;
        ostringstream ost;
        ost << setprecision(3) << promedioErrorPorMes(i) << " " << desvioErrorPorMes(i);
        istringstream ist{ost.str()};
        ist >> errorStr >> desvioStr;

        gp << R"(set title ")" << i + 1 << R"( meses hacia adelante\n)"
           << R"(EARP = )" << errorStr
           << R"( %, Desvío = )" << desvioStr
           << R"( %")"
           << " font ',11'" << endl
           << "plot " << gp.file1d(ventas(span(i, i + salidasArima.n_rows - 1)).eval()) << " with linespoints title 'Salida original', "
           << gp.file1d(salidasArima.col(i).eval()) << " with linespoints title 'Salida de la red' lw 2" << endl;
    }

    getchar();

    return 0;
}
