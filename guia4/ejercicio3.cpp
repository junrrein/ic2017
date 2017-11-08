#include "costo_uniforme.cpp"
#include "colonia_hormigas.cpp"
#include "../config.hpp"

int main()
{
    arma_rng::set_seed_random();

    const string rutaArchivo7 = config::sourceDir + "/guia4/datos/7cities.csv";
    //    ArbolBusqueda a7{rutaArchivo7};

    //    cout << a7.hacerBusqueda();

    //    const string rutaArchivo10 = config::sourceDir + "/guia4/datos/10cities.csv";
    //    ArbolBusqueda a10{rutaArchivo10};

    //    cout << a10.hacerBusqueda();

    ColoniaHormigas c7(rutaArchivo7,
                       /*nHormigas =*/20,
                       /*sigma_cero =*/0.3,
                       /*alpha =*/2,
                       /*beta =*/2,
                       /*Q =*/1);

    cout << c7.seleccionarVecino(1, {2, 3, 4, 5, 6, 7}) << endl;

    return 0;
}
