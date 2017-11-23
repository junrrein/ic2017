#include "construir_tuplas.cpp"
#include "../../guia2/mlp_salida_lineal.cpp"
#include "../../config.hpp"
#include <gnuplot-iostream.h>
#include <sstream>

using namespace ic;

int main()
{
    const string rutaBase = config::sourceDir + "/TP_Final/datos/";
    const string rutaDiferencias = rutaBase + "DiferenciasRelativas.csv";

    const int nRetardos = 9;
    const int nSalidas = 3;
    ParametrosMulticapa parametros;
    parametros.estructuraRed = {24, nSalidas};
    parametros.nEpocas = 2000;
    parametros.tasaAprendizaje = 0.00075;
    parametros.inercia = 0.2;
    parametros.toleranciaError = 10;

    // Carga de datos
    Particion particion = cargarTuplas({rutaDiferencias},
                                       rutaDiferencias,
                                       nRetardos,
                                       nSalidas);

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

    // Calculando las salidas para los datos de prueba
    vector<vec> salidaRed(nSalidas, vec(particion.prueba.tuplasEntrada.n_rows));

    for (unsigned int n = 0; n < particion.prueba.tuplasEntrada.n_rows; ++n) {
        vec salidas = ic::salidaMulticapa(pesos,
                                          particion.prueba.tuplasEntrada.row(n).t())
                          .back();

        for (int j = 0; j < nSalidas; ++j)
            salidaRed.at(j)(n) = salidas(j);
    }

    // Desnormalización
    vec diferencias;
    diferencias.load(rutaDiferencias);

    for (unsigned int i = 0; i < salidaRed.size(); ++i) {
        salidaRed.at(i) = desnormalizar(diferencias, salidaRed.at(i));
        particion.prueba.tuplasSalida.col(i) = desnormalizar(diferencias, particion.prueba.tuplasSalida.col(i));
    }

    // Pasar diferencias relativas a patentamientos en unidades

    // Carga de datos de patentamientos en unidades
    const string rutaVentas = rutaBase + "Ventas.csv";
    vec ventas;
    ventas.load(rutaVentas);
    const vec ventasPrueba = ventas.tail(16);

    // Pasar datos de diferencias relativas a patentamientos en unidades
    for (unsigned int i = 0; i < salidaRed.size(); ++i) {
        for (unsigned int j = 0; j < salidaRed.front().n_elem; ++j) {
            if (i == 0) {
                salidaRed.at(i)(j) = salidaRed.at(i)(j) / 100
                                         * ventasPrueba(nRetardos + i + j - 1)
                                     + ventasPrueba(nRetardos + i + j - 1);
            }
            else {
                salidaRed.at(i)(j) = salidaRed.at(i)(j) / 100
                                         * salidaRed.at(i - 1)(j)
                                     + salidaRed.at(i - 1)(j);
            }

            particion.prueba.tuplasSalida(j, i) = particion.prueba.tuplasSalida(j, i) / 100
                                                      * ventasPrueba(nRetardos + i + j - 1)
                                                  + ventasPrueba(nRetardos + i + j - 1);
        }
    }

    // Cálculo de errror
    vec promedioErrorPorMes(nSalidas);
    vec desvioErrorPorMes(nSalidas);
    for (int i = 0; i < nSalidas; ++i) {
        const vec erroresRelativosAbsolutos = abs(particion.prueba.tuplasSalida.col(i) - salidaRed.at(i))
                                              / particion.prueba.tuplasSalida.col(i)
                                              * 100;
        promedioErrorPorMes(i) = mean(erroresRelativosAbsolutos);
        desvioErrorPorMes(i) = stddev(erroresRelativosAbsolutos);
    }

    // Guardar errores a un archivo para comparar distintos modelos
    const mat errores = join_horiz(promedioErrorPorMes, desvioErrorPorMes);
    errores.save(rutaBase + "errorMlpSalidaDifRel.csv", arma::csv_ascii);

    Gnuplot gp;
    gp << "set terminal qt size 600,700" << endl
       << "set multiplot layout 3,1 title 'Predicción usando Diferencias Relativas de Patentamientos - Red MLP' font ',13'" << endl
       << "set xlabel 'Mes (final de la serie)'" << endl
       << "set ylabel 'Patentamientos (unidades)'" << endl
       << "set yrange [0:90000]" << endl
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
