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

    ParametrosMulticapa parametros;
    parametros.estructuraRed = {23, 6};
    parametros.nEpocas = 2000;
    parametros.tasaAprendizaje = 0.00075;
    parametros.inercia = 0.2;
    parametros.toleranciaError = 10;

    // Carga de datos
    Particion particion = cargarTuplas({rutaDiferencias},
                                       rutaDiferencias,
                                       13,
                                       6,
                                       true);

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
    vector<vec> salidaRed(6, vec(particion.prueba.tuplasEntrada.n_rows));

    for (unsigned int n = 0; n < particion.prueba.tuplasEntrada.n_rows; ++n) {
        vec salidas = ic::salidaMulticapa(pesos,
                                          particion.prueba.tuplasEntrada.row(n).t())
                          .back();

        for (int j = 0; j < 6; ++j)
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
    Particion particionAux = cargarTuplas({rutaVentas},
                                          rutaVentas,
                                          13,
                                          6,
                                          true);
    mat tuplasVentas = join_vert(particionAux.evaluacion.tuplasSalida.row(particionAux.evaluacion.tuplasSalida.n_rows - 1),
                                 particionAux.prueba.tuplasSalida);

    if (tuplasVentas.n_rows != salidaRed.front().n_elem + 1)
        throw runtime_error("Acá están mal la cantidad de tuplas");

    // Normalización de los patentamientos
    vec ventas;
    ventas.load(rutaVentas);
    tuplasVentas.each_col([&](vec& v) {
        v = desnormalizar(ventas, v);
    });

    for (unsigned int i = 0; i < salidaRed.size(); ++i) {
        for (unsigned int j = 0; j < salidaRed.front().n_elem; ++j) {
            salidaRed.at(i)(j) = salidaRed.at(i)(j) / 100
                                     * tuplasVentas(j, i)
                                 + tuplasVentas(j, i);
            particion.prueba.tuplasSalida(j, i) = particion.prueba.tuplasSalida(j, i) / 100
                                                      * tuplasVentas(j, i)
                                                  + tuplasVentas(j, i);
        }
    }

    // Cálculo de errror
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
    errores.save(rutaBase + "errorMlpSalidaDifRel.csv", arma::csv_ascii);

    Gnuplot gp;
    gp << "set terminal qt size 1200,600" << endl
       << "set multiplot layout 2,3 title 'Predicción usando Diferencia Relativa de Patentamientos' font ',12'" << endl
       << "set xlabel 'Mes (final de la serie)'" << endl
       << "set ylabel 'Diferencias relativas (%)'" << endl
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
