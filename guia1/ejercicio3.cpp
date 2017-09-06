#include "multicapa.cpp"
#include "particionar.cpp"
#include "../config.hpp"
#include <gnuplot-iostream.h>

int main()
{
    arma_rng::set_seed_random();

    mat datos;
    datos.load(config::sourceDir + "/guia1/icgtp1datos/concentlite.csv");
    //string rutaCarpeta = config::sourceDir + "/guia1/icgtp1datos/particionesConcent/";
    //const vector<ic::Particion> particiones = ic::cargarParticiones(rutaCarpeta, 10);

    ifstream ifs{config::sourceDir + "/guia1/parametrosConcentMulticapa.txt"};
    ic::ParametrosMulticapa parametros;
    if (!(ifs >> parametros))
        throw runtime_error{"Error al leer los parámetros"};

    vector<mat> pesos;
    double tasaError;
    int epoca;

    tie(pesos, tasaError, epoca) = ic::entrenarMulticapa(parametros.estructuraRed,
                                                         datos,
                                                         parametros.nEpocas,
                                                         parametros.tasaAprendizaje,
                                                         parametros.parametroSigmoidea,
                                                         parametros.toleranciaError);

    cout << "Multicapa sin inercia" << endl
         << "Tasa de error: " << tasaError << endl
         << "N° de epocas de entrenamiento: " << epoca << endl;

    mat falsosPositivos, falsosNegativos, verdaderosPositivos, verdaderosNegativos;

    for (unsigned int i = 0; i < datos.n_rows; i++) {
        const rowvec patron = datos.row(i).head(2);
        const double salidaDeseada = datos(i, 2);

        vec salidaRed = ic::salidaMulticapa(pesos, patron.t()).back();
        salidaRed = ic::pendorcho(salidaRed);

        if (salidaRed(0) == salidaDeseada) {
            if (salidaDeseada == 1)
                verdaderosPositivos.insert_rows(verdaderosPositivos.n_rows, patron);
            else
                verdaderosNegativos.insert_rows(verdaderosNegativos.n_rows, patron);
        }
        else {
            if (salidaDeseada == 1)
                falsosPositivos.insert_rows(falsosPositivos.n_rows, patron);
            else
                falsosNegativos.insert_rows(falsosNegativos.n_rows, patron);
        }
    }

    // Parte b: Experimentar qué pasa al usar el término de inercia
    // en el algoritmo de ajuste de pesos

    ifs.close();
    ifs.open(config::sourceDir + "/guia1/parametrosConcentInercia.txt");
    if (!(ifs >> parametros))
        throw runtime_error{"Error al leer los parámetros"};

    tie(pesos, tasaError, epoca) = ic::entrenarMulticapa(parametros.estructuraRed,
                                                         datos,
                                                         parametros.nEpocas,
                                                         parametros.tasaAprendizaje,
                                                         parametros.inercia,
                                                         parametros.parametroSigmoidea,
                                                         parametros.toleranciaError);

    cout << "Multicapa con inercia " << parametros.inercia << endl
         << "Tasa de error: " << tasaError << endl
         << "N° de epocas de entrenamiento: " << epoca << endl;

    //    Gnuplot gp;

    //    gp << "set title 'Clasificación de patrones' font ',13'\n"
    //       << "set xlabel 'x_1'\n"
    //       << "set ylabel 'x_2'\n"
    //       << "set grid\n"
    //       << "set pointsize 2\n"
    //       << "plot " << gp.file1d(verdaderosPositivos) << "title 'Verdaderos Positivos' with points lt rgb 'blue', "
    //       << gp.file1d(verdaderosNegativos) << "title 'Verdaderos Negativos' with points lt rgb 'red', "
    //       << gp.file1d(falsosPositivos) << "title 'Falsos Positivos' with points lt rgb 'red' ps 1.5 pt 5, "
    //       << gp.file1d(falsosNegativos) << "title 'Falsos Negativos' with points lt rgb 'blue' ps 1.5 pt 5" << endl;

    //    getchar();

    return 0;
}
