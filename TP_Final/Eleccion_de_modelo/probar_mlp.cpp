#include "construir_tuplas.cpp"
#include "../../guia2/mlp_salida_lineal.cpp"
#include "../../config.hpp"
#include <gnuplot-iostream.h>
#include <sstream>

using namespace ic;

int main()
{
    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    const string rutaVentas = rutaBase + "Ventas.csv";
    const string rutaDiferencias = rutaBase + "DiferenciasRelativas.csv";
    const string rutaExportaciones = rutaBase + "Exportaciones.csv";

    ParametrosMulticapa parametros;
    parametros.estructuraRed = {23, 6};
    parametros.nEpocas = 2000;
    parametros.tasaAprendizaje = 0.00075;
    parametros.inercia = 0.2;
    parametros.toleranciaError = 10;

    // Carga de datos
    Particion particion = cargarTuplas({rutaDiferencias, rutaExportaciones},
                                       rutaVentas,
                                       14,
                                       6,
                                       false);

    const mat datosEntrenamiento = join_vert(join_horiz(particion.entrenamiento.tuplasEntrada,
                                                        particion.entrenamiento.tuplasSalida),
                                             join_horiz(particion.evaluacion.tuplasEntrada,
                                                        particion.evaluacion.tuplasSalida));

    // Entrenamiento
    vector<mat> pesos;
    tie(pesos, ignore, ignore) = entrenarMulticapa(parametros.estructuraRed,
                                                   datosEntrenamiento,
                                                   parametros.nEpocas,
                                                   parametros.tasaAprendizaje,
                                                   parametros.inercia,
                                                   parametros.toleranciaError,
                                                   true);

    // Prueba
    // Calcular la salida para los datos de prueba
    vector<vec> salidaRed(6, vec(particion.prueba.tuplasEntrada.n_rows));

    for (unsigned int n = 0; n < particion.prueba.tuplasEntrada.n_rows; ++n) {
        vec salidas = ic::salidaMulticapa(pesos,
                                          particion.prueba.tuplasEntrada.row(n).t())
                          .back();

        for (int j = 0; j < 6; ++j)
            salidaRed.at(j)(n) = salidas(j);
    }

    // Desnormalizar datos
    vec ventas;
    ventas.load(rutaVentas);

    for (unsigned int i = 0; i < salidaRed.size(); ++i) {
        salidaRed.at(i) = desnormalizar(ventas, salidaRed.at(i));
        particion.prueba.tuplasSalida.col(i) = desnormalizar(ventas, particion.prueba.tuplasSalida.col(i));
    }

    // Calculo de error
    vec promedioErrorPorMes(6);
    vec desvioErrorPorMes(6);
    for (int i = 0; i < 6; ++i) {
        const vec erroresRelativosAbsolutos = abs(particion.prueba.tuplasSalida.col(i) - salidaRed.at(i))
                                              / particion.prueba.tuplasSalida.col(i)
                                              * 100;
        promedioErrorPorMes(i) = mean(erroresRelativosAbsolutos);
        desvioErrorPorMes(i) = stddev(erroresRelativosAbsolutos);
    }

    // Guardar errores a un archivo para comparar distintos modelos
    const mat errores = join_horiz(promedioErrorPorMes, desvioErrorPorMes);
    errores.save(rutaBase + "errorMlpSalidaVentas.csv", arma::csv_ascii);

    Gnuplot gp;
    gp << "set terminal qt size 1200,600" << endl
       << "set multiplot layout 2,3 title 'Predicción usando Diferencias Relativas de Patentamientos + Exportaciones - Red MLP' font ',12'" << endl
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
           << "plot " << gp.file1d(particion.prueba.tuplasSalida.col(0).eval()) << " with linespoints title 'Salida original', "
           << gp.file1d(salidaRed.at(0)) << " with linespoints title 'Salida de la red' lw 2" << endl;
    }

    for (unsigned int i = 1; i < salidaRed.size(); ++i) {
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
           << "plot " << gp.file1d(particion.prueba.tuplasSalida.col(i).eval()) << " with linespoints title 'Salida original', "
           << gp.file1d(salidaRed.at(i)) << " with linespoints title 'Salida de la red' lw 2" << endl;
    }

    getchar();

    return 0;
}
