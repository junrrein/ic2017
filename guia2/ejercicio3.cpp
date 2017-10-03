#include "som.cpp"
#include "../config.hpp"

int main()
{
    arma_rng::set_seed_random();

    // -----------------------------
    //        circulo.csv
    // -----------------------------
    mat datos;
    datos.load(config::sourceDir + "/guia2/datos/circulo.csv");

    // Mapa 2D
    ic::SOM somCirculo2D{datos, {5, 4}};
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
    ic::SOM somCirculo1D{datos, {20, 1}};

    somCirculo1D.entrenar(500, 0.7, 0.7, 5, 5);
    gp << "set title 'SOM 1D para circulo.csv - Etapa de ordenamiento topológico'" << endl;
    somCirculo1D.graficar(gp);
    getchar();

    somCirculo1D.entrenar(1000, 0.7, 0.1, 7, 1);
    gp << "set title 'SOM 1D para circulo.csv - Etapa de transición'" << endl;
    somCirculo1D.graficar(gp);
    getchar();

    somCirculo1D.entrenar(3000, 0.01, 0.01, 0, 0);
    gp << "set title 'SOM 1D para circulo.csv - Etapa de ajuste fino'" << endl;
    somCirculo1D.graficar(gp);
    getchar();

    // -----------------------------
    //           te.csv
    // -----------------------------
    datos.load(config::sourceDir + "/guia2/datos/te.csv");

    // Mapa 2D
    ic::SOM somTe2D{datos, {2, 8}};

    somTe2D.entrenar(500, 0.7, 0.7, 2, 2);
    gp << "set title 'SOM 2D para te.csv - Etapa de ordenamiento topológico'" << endl
       << "set xrange [-1.5:1.5]" << endl
       << "set yrange [-1.5:1.5]" << endl;
    somTe2D.graficar(gp);
    getchar();

    somTe2D.entrenar(1000, 0.7, 0.1, 2, 1);
    gp << "set title 'SOM 2D para te.csv - Etapa de transición'" << endl;
    somTe2D.graficar(gp);
    getchar();

    somTe2D.entrenar(3000, 0.01, 0.01, 0, 0);
    gp << "set title 'SOM 2D para te.csv - Etapa de ajuste fino'" << endl;
    somTe2D.graficar(gp);
    getchar();

    // Mapa 1D
    ic::SOM somTe1D{datos, {16, 1}};

    somTe1D.entrenar(500, 0.7, 0.7, 4, 4);
    gp << "set title 'SOM 1D para te.csv - Etapa de ordenamiento topológico'" << endl;
    somTe1D.graficar(gp);
    getchar();

    somTe1D.entrenar(1000, 0.7, 0.1, 4, 1);
    gp << "set title 'SOM 1D para te.csv - Etapa de transición'" << endl;
    somTe1D.graficar(gp);
    getchar();

    somTe1D.entrenar(3000, 0.01, 0.01, 0, 0);
    gp << "set title 'SOM 1D para te.csv - Etapa de ajuste fino'" << endl;
    somTe1D.graficar(gp);
    getchar();

    return 0;
}
