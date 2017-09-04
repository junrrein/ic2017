#include <iostream>
#include <sstream>
#include "multicapa.cpp"
#include "../config.hpp"

using namespace std;
using namespace arma;

int main()
{
    arma_rng::set_seed_random();

    ifstream ifs{config::sourceDir + "/guia1/xorParametrosEntrenamiento.txt"};
    ic::ParametrosMulticapa parametros;
    if (!(ifs >> parametros))
        throw runtime_error("No se pudo cargar correctamente el archivo de parámetros");

    mat datos;
    datos.load(config::sourceDir + "/guia1/icgtp1datos/XOR_trn.csv");

    vector<mat> pesos;
    double tasaError;
    tie(pesos, tasaError) = ic::entrenarMulticapa(parametros.estructuraRed,
                                                  datos,
                                                  parametros.nEpocas,
                                                  parametros.tasaAprendizaje,
                                                  parametros.parametroSigmoidea,
                                                  parametros.toleranciaError);

    cout << "Tasa de error del Multicapa [2 1] para el XOR (entrenamiento): " << tasaError << endl;

    datos.load(config::sourceDir + "/guia1/icgtp1datos/XOR_tst.csv");
    const mat patrones = datos.head_cols(2);
    const mat salidaDeseada = datos.tail_cols(1);
    tasaError = ic::errorPrueba(pesos, patrones, salidaDeseada, parametros.toleranciaError);

    cout << "Tasa de error del Multicapa [2 1] para el XOR (prueba): " << tasaError << endl;

    return 0;
}
