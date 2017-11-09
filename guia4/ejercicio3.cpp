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
                       /*nHormigas =*/100,
                       /*nEpocas =*/500,
                       /*sigma_cero =*/0.5,
                       /*alpha =*/4,
                       /*beta =*/4,
                       /*tasaEvaporacion =*/0.1,
                       /*Q =*/2);

    cout << c7.encontrarSolucion() << endl;

    return 0;
}
