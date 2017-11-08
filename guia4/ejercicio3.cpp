#include "costo_uniforme.cpp"
#include "../config.hpp"

int main()
{
    arma_rng::set_seed_random();
    srand(time(0));

    const string rutaArchivo7 = config::sourceDir + "/guia4/datos/7cities.csv";
    ArbolBusqueda a7{rutaArchivo7};

    cout << a7.hacerBusqueda();

    const string rutaArchivo10 = config::sourceDir + "/guia4/datos/10cities.csv";
    ArbolBusqueda a10{rutaArchivo10};

    cout << a10.hacerBusqueda();

    return 0;
}
