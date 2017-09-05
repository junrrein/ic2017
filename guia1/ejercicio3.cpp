#include "multicapa.cpp"
#include "particionar.cpp"
#include "../config.hpp"

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

    cout << tasaError << endl
         << epoca << endl;

    return 0;
}
