#include "construir_tuplas.cpp"
#include "../../guia2/mlp_salida_lineal.cpp"
#include "../../guia2/radial_por_lotes.cpp"
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
    ParametrosRBF parametros;
    parametros.estructuraRed = {22, nSalidas};
    parametros.nEpocas = 2000;
    parametros.tasaAprendizaje = 0.00075;
    parametros.inercia = 0.2;
    parametros.toleranciaError = 10;

    // Carga de datos
    Particion particion = cargarTuplas({rutaDiferencias},
                                       rutaDiferencias,
                                       nRetardos,
                                       nSalidas);

    const mat datosEntrenamiento = join_vert(particion.entrenamiento.tuplasEntrada,
                                             particion.evaluacion.tuplasEntrada);

    // Entrenamiento
    vector<rowvec> centroides;
    vec sigmas;
    tie(centroides, sigmas)
        = entrenarRadialPorLotes(datosEntrenamiento,
                                 parametros.estructuraRed(0),
                                 tipoInicializacion::patronesAlAzar,
                                 0.2);

    mat salidasRadiales(datosEntrenamiento.n_rows,
                        centroides.size());
    for (unsigned int j = 0; j < datosEntrenamiento.n_rows; ++j)
        salidasRadiales.row(j) = salidaRadial(datosEntrenamiento.row(j),
                                              centroides,
                                              sigmas);

    mat datosCapaFinal = join_horiz(salidasRadiales,
                                    join_vert(particion.entrenamiento.tuplasSalida,
                                              particion.evaluacion.tuplasSalida));

    vector<mat> pesos;
    tie(pesos, ignore, ignore) = entrenarMulticapa(vec{parametros.estructuraRed(1)},
                                                   datosCapaFinal,
                                                   parametros.nEpocas,
                                                   parametros.tasaAprendizaje,
                                                   parametros.inercia,
                                                   parametros.toleranciaError,
                                                   true);

    // Prueba
    // Calcular la salida para los datos de prueba
    salidasRadiales = mat(particion.prueba.tuplasEntrada.n_rows,
                          centroides.size());
    for (unsigned int j = 0; j < particion.prueba.tuplasEntrada.n_rows; ++j)
        salidasRadiales.row(j) = salidaRadial(particion.prueba.tuplasEntrada.row(j),
                                              centroides,
                                              sigmas);

    vector<vec> salidaRed(nSalidas, vec(salidasRadiales.n_rows));

    for (unsigned int n = 0; n < salidasRadiales.n_rows; ++n) {
        vec salidas = ic::salidaMulticapa(pesos,
                                          salidasRadiales.row(n).t())
                          .back();

        for (unsigned int j = 0; j < salidaRed.size(); ++j)
            salidaRed.at(j)(n) = salidas(j);
    }

    // Desnormalizar datos
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

    // Calculo de error
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
    errores.save(rutaBase + "errorRbfSalidaDifRel.csv", arma::csv_ascii);

    Gnuplot gp;
    gp << "set terminal qt size 600,700" << endl
       << "set multiplot layout 3,1 title 'Predicción usando Diferencias Relativas de Patentamientos - Red con RBF' font ',12'" << endl
       << "set xlabel 'Mes (final de la serie)'" << endl
       << "set ylabel 'Patentamientos (unidades)'" << endl
       << "set yrange [0:200000]" << endl
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
