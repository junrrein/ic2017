#include "multicapa.cpp"
#include "particionar.cpp"
#include "../config.hpp"

int main()
{
    arma_rng::set_seed_random();

    mat datos;
    string rutaBaseDatos = config::sourceDir + "/guia1/icgtp1datos/";
    datos.load(rutaBaseDatos + "irisbin.csv");

    ifstream ifs{config::sourceDir + "/guia1/parametrosIris.txt"};
    ic::ParametrosMulticapa parametros;
    if (!(ifs >> parametros))
        throw runtime_error("No se pudo cargar el archivo de parámetros");

    // Leave K Out

    {
        vector<ic::Particion> particiones = ic::cargarParticiones(rutaBaseDatos + "particionesIrisKOut/", 10);
        vec errores;
        errores.resize(particiones.size());
        int noConvergio = 0;

        for (unsigned int i = 0; i < particiones.size(); ++i) {
            vector<mat> pesos;
            int epocas;
            tie(pesos, ignore, epocas) = ic::entrenarMulticapa(parametros.estructuraRed,
                                                               datos.rows(particiones[i].first),
                                                               parametros.nEpocas,
                                                               parametros.tasaAprendizaje,
                                                               parametros.inercia,
                                                               parametros.parametroSigmoidea,
                                                               parametros.toleranciaError);

            if (epocas == parametros.nEpocas)
                ++noConvergio;

            const double tasaError = ic::errorPrueba(pesos,
                                                     datos.rows(particiones[i].second),
                                                     parametros.parametroSigmoidea);
            errores[i] = tasaError;
        }

        cout << "Iris multicapa, Leave K Out" << endl
             << "Error promedio: " << mean(errores) << endl
             << "Desvío estándar del error: " << stddev(errores) << endl
             << "El algoritmo no convergió en " << noConvergio << " casos" << endl;
    }

    // Leave One Out

    {
        vector<ic::Particion> particiones = ic::cargarParticiones(rutaBaseDatos + "particionesIris1Out/", 150);
        vec errores;
        errores.resize(particiones.size());
        int noConvergio = 0;

        for (unsigned int i = 0; i < particiones.size(); ++i) {
            vector<mat> pesos;
            int epocas;
            tie(pesos, ignore, epocas) = ic::entrenarMulticapa(parametros.estructuraRed,
                                                               datos.rows(particiones[i].first),
                                                               parametros.nEpocas,
                                                               parametros.tasaAprendizaje,
                                                               parametros.inercia,
                                                               parametros.parametroSigmoidea,
                                                               parametros.toleranciaError);

            if (epocas == parametros.nEpocas)
                ++noConvergio;

            const double tasaError = ic::errorPrueba(pesos,
                                                     datos.rows(particiones[i].second),
                                                     parametros.parametroSigmoidea);
            errores[i] = tasaError;
        }

        cout << "\nIris multicapa, Leave One Out" << endl
             << "Error promedio: " << mean(errores) << endl
             << "Desvío estándar del error: " << stddev(errores) << endl
             << "El algoritmo no convergió en " << noConvergio << " casos" << endl;
    }

    return 0;
}
