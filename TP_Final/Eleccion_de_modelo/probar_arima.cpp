#include <armadillo>
#include <gnuplot-iostream.h>
#include "../../config.hpp"

using namespace arma;
using namespace std;

int main()
{
    const string rutaBase = config::sourceDir + "/TP_Final/";

    const vec ventas = {44305, 37316, 23008, 42900, 36788, 43908, 45160, 44184, 39445, 48209, 54326, 52254, 45313, 41938, 29057, 68485, 42320, 57054, 48130};
    mat salidasArima = {{39052, 35792, 22466, 66102, 38072, 43850},
                        {37242, 25970, 68612, 40750, 46332, 45686},
                        {26022, 68654, 40818, 46384, 45745, 47947},
                        {68034, 38754, 45010, 44128, 46498, 46003},
                        {29752, 28482, 31876, 31410, 33003, 36384},
                        {31194, 36549, 35208, 36842, 40221, 37420},
                        {41909, 43295, 44573, 47462, 45067, 49386},
                        {44631, 46783, 49232, 46911, 51241, 47220},
                        {46594, 48917, 46644, 50959, 46939, 40869},
                        {45920, 41800, 46888, 42608, 36725, 22919},
                        {42761, 48465, 43959, 37995, 24187, 59779},
                        {52982, 51184, 43868, 30237, 66119, 44880},
                        {50864, 43433, 29883, 65804, 44530, 49277},
                        {41345, 26843, 63017, 41838, 46386, 46347}};

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
    errores.save(rutaBase + "datos/errorArimaSalidaVentas.csv", arma::csv_ascii);

    Gnuplot gp;
    gp << "set terminal qt size 1200,600" << endl
       << "set multiplot layout 2,3 title 'Predicción usando Patentamientos - ARIMA' font ',12'" << endl
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
