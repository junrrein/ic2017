#include "som.cpp"
#include "../config.hpp"

int main()
{
    arma_rng::set_seed_random();

    mat datos;
    datos.load(config::sourceDir + "/guia2/datos/clouds.csv");

    ic::SOM somClouds{datos.head_cols(2), {5, 5}};
    Gnuplot gp;

    // Etapa de ordenamiento topológico
    for (int i = 1; i <= 3; ++i) {
        somClouds.entrenar(1, 0.9, 0.9, 2, 2);
        gp << "set title 'SOM para clouds.csv - Etapa de ordenamiento topológico - " << 1 * i << " épocas' font ',12'" << endl;
        somClouds.graficar(gp);
        getchar();
    }

    // Etapa de transición
    {
        const int nEpocas = 100;
        const vec velocidad = linspace(0.9, 0.1, nEpocas);
        const vec vecindad = round(linspace(2, 1, nEpocas));
        for (int i = 0; i < 5; ++i) {
            somClouds.entrenar(20, velocidad(i), velocidad(i), vecindad(i), vecindad(i));
            gp << "set title 'SOM para clouds.csv - Etapa de transición - " << 20 * (i + 1) << " épocas'" << endl;
            cout << "Velocidad: " << to_string(velocidad(i)) << " - Vecindad: " << vecindad(i) << endl;
            somClouds.graficar(gp);
            getchar();
        }
    }

    // Etapa de ajuste fino
    for (int i = 1; i <= 12; ++i) {
        somClouds.entrenar(5, 0.01, 0.01, 0, 0);
        gp << "set title 'SOM para clouds.csv - Etapa de ajuste fino - " << 5 * i << " épocas'" << endl;
        somClouds.graficar(gp);
        getchar();
    }

    return 0;
}
