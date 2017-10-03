#include "som.cpp"
#include "../config.hpp"

int main()
{
    arma_rng::set_seed_random();

    mat datos;
    datos.load(config::sourceDir + "/guia2/datos/circulo.csv");

    // Mapa 2D
    SOM somCirculo2D{datos, {5, 6}};
    Gnuplot gp;

    somCirculo2D.entrenar(500, 0.7, 0.7, 2, 2);
    gp << "set title 'SOM 2D para circulo.csv - Etapa de ordenamiento topológico' font ',12'" << endl;
    somCirculo2D.graficar(gp);
    getchar();

    somCirculo2D.entrenar(1000, 0.7, 0.1, 2, 1);
    gp << "set title 'SOM 2D para circulo.csv - Etapa de transición'" << endl;
    somCirculo2D.graficar(gp);
    getchar();

    somCirculo2D.entrenar(3000, 0.01, 0.01, 0, 0);
    gp << "set title 'SOM 2D para circulo.csv - Etapa de ajuste fino'" << endl;
    somCirculo2D.graficar(gp);
    getchar();

    // Mapa 1D
    SOM somCirculo1D{datos, {30, 1}};

    somCirculo1D.entrenar(500, 0.7, 0.7, 2, 2);
    gp << "set title 'SOM 1D para circulo.csv - Etapa de ordenamiento topológico' font ',12'" << endl;
    somCirculo1D.graficar(gp);
    getchar();

    somCirculo1D.entrenar(1000, 0.7, 0.1, 2, 1);
    gp << "set title 'SOM 1D para circulo.csv - Etapa de transición'" << endl;
    somCirculo1D.graficar(gp);
    getchar();

    somCirculo1D.entrenar(3000, 0.01, 0.01, 0, 0);
    gp << "set title 'SOM 1D para circulo.csv - Etapa de ajuste fino'" << endl;
    somCirculo1D.graficar(gp);
    getchar();

    return 0;
}
