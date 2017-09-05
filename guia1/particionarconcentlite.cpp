#include <boost/filesystem.hpp>
#include "particionar.cpp"
#include "../config.hpp"

int main()
{
    mat datos;
    datos.load(config::sourceDir + "/guia1/icgtp1datos/concentlite.csv");

    auto particiones = ic::particionar(datos, 10, 80);
    string carpetaParticiones = config::sourceDir + "/guia1/icgtp1datos/particionesConcent/";
    boost::filesystem::create_directory(carpetaParticiones);
    ic::guardarParticiones(particiones, carpetaParticiones);

    return 0;
}
