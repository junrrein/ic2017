#include <boost/filesystem.hpp>
#include "particionar.cpp"
#include "../config.hpp"

int main()
{
    arma_rng::set_seed_random();

    mat datos;
    datos.load(config::sourceDir + "/guia1/icgtp1datos/irisbin.csv");

    // Leave K out

    vector<ic::Particion> particiones = ic::leaveKOut(datos, 15);
    string carpetaParticiones = config::sourceDir + "/guia1/icgtp1datos/particionesIrisKOut/";
    boost::filesystem::create_directory(carpetaParticiones);
    ic::guardarParticiones(particiones, carpetaParticiones);

    // Leave one out

    particiones = ic::leaveKOut(datos, 1);
    carpetaParticiones = config::sourceDir + "/guia1/icgtp1datos/particionesIris1Out/";
    boost::filesystem::create_directory(carpetaParticiones);
    ic::guardarParticiones(particiones, carpetaParticiones);

    return 0;
}
