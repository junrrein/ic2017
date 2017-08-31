#include <iostream>
#include "multicapa.cpp"

using namespace std;
using namespace arma;

int main()
{
    arma_rng::set_seed_random();
    mat datos;
    datos.load("XOR_trn.csv");

    vector<mat> pesos;
    double tasaError;
    const ic::EstructuraCapasRed estructura = {2, 1};
    tie(pesos, tasaError) = ic::entrenarMulticapa(estructura, datos, 200, 0.4, 1, 1);

    cout << "Tasa de error del Multicapa [2 1] para el XOR (entrenamiento): " << tasaError << endl;

    datos.load("XOR_tst.csv");
    const mat patrones = datos.head_cols(2);
    const mat salidaDeseada = datos.tail_cols(1);
    tasaError = ic::errorPrueba(pesos, patrones, salidaDeseada, 1);

    cout << "Tasa de error del Multicapa [2 1] para el XOR (prueba): " << tasaError << endl;

    return 0;
}
