#include "som.cpp"
#include "../config.hpp"

int main()
{
    arma_rng::set_seed_random();

    mat datos;
    datos.load(config::sourceDir + "/guia2/datos/clouds.csv");
    const mat patrones = datos.head_cols(2);
    const vec salidaDeseada = datos.tail_cols(1);

    Gnuplot gp;
    ic::SOM somClouds{patrones, {9, 9}, salidaDeseada};

    // Etapa de ordenamiento topológico
    somClouds.entrenar(1000, 0.7, 0.7, 2, 2);
    gp << "set title 'SOM para clouds.csv - Etapa de ordenamiento topológico' font ',12'" << endl;
    somClouds.graficar(gp);
    getchar();

    // Etapa de transición
    somClouds.entrenar(1000, 0.7, 0.1, 2, 1);
    gp << "set title 'SOM para clouds.csv - Etapa de transición'" << endl;
    somClouds.graficar(gp);
    getchar();

    // Etapa de ajuste fino
    somClouds.entrenar(3000, 0.01, 0.01, 0, 0);
    gp << "set title 'SOM para clouds.csv - Etapa de ajuste fino'" << endl;
    somClouds.graficar(gp);
    getchar();

    // Etiquetado
    somClouds.etiquetar();

    // Clasificar patrones según el SOM
    vec clasificacion = somClouds.clasificar(patrones);

    // Clasificar patrones en verdaderos y falsos, positivos y negativos
    mat verdaderosPositivos, verdaderosNegativos, falsosPositivos, falsosNegativos;

    for (unsigned int n = 0; n < patrones.n_rows; ++n) {
        if (clasificacion(n) == salidaDeseada(n)) { // Tenemos un verdadero
            if (salidaDeseada(n) == 0)
                verdaderosPositivos.insert_rows(verdaderosPositivos.n_rows, patrones.row(n));
            else
                verdaderosNegativos.insert_rows(verdaderosNegativos.n_rows, patrones.row(n));
        }
        else { // Tenemos un falso
            if (salidaDeseada(n) == 0)
                falsosNegativos.insert_rows(falsosNegativos.n_rows, patrones.row(n));
            else
                falsosPositivos.insert_rows(falsosPositivos.n_rows, patrones.row(n));
        }
    }

    // Clasificar neuronas según su etiqueta
    const field<rowvec>& mapa = somClouds.mapa();
    const mat etiquetas = somClouds.etiquetas();

    const double error = double(verdaderosPositivos.n_rows + verdaderosNegativos.n_rows)
                         / (falsosPositivos.n_rows + falsosNegativos.n_rows);
    ostringstream ost;
    ost << setprecision(2) << error;

    // Graficar lo etiquetado y clasificado
    Gnuplot gp2;
    gp2 << "set title 'SOM para clouds.csv - Clasificación - Error: " + ost.str() + " %' font ',12'" << endl
        << "set key box opaque width 3" << endl
        << "set xlabel 'x_1' font ',11'" << endl
        << "set ylabel 'x_2' font ',11'" << endl

        // Graficar patrones
        << "plot " << gp2.file1d(verdaderosPositivos) << "title 'Verdaderos Positivos' with points pt 2 ps 1 lt rgb 'blue', "
        << gp2.file1d(verdaderosNegativos) << "title 'Verdaderos Negativos' with points pt 2 ps 1 lt rgb 'orange', "
        << gp2.file1d(falsosPositivos) << "title 'Falsos Positivos' with points pt 4 ps 1 lt rgb 'orange', "
        << gp2.file1d(falsosNegativos) << "title 'Falsos Negativos' with points pt 4 ps 1 lt rgb 'blue', ";

    // Graficar neuronas del mapa y las conexiones
    for (unsigned int x = 0; x < mapa.n_rows; ++x) {
        for (unsigned int y = 0; y < mapa.n_cols; ++y) {
            // Graficar la neurona
            if (etiquetas(x, y) == 0)
                gp2 << gp2.file1d(mapa(x, y).eval()) << "notitle with points ps 2 pt 6 lw 3 lt rgb 'cyan', ";
            else
                gp2 << gp2.file1d(mapa(x, y).eval()) << "notitle with points ps 2 pt 6 lw 3 lt rgb 'red', ";
        }
    }

    // Título de los centroides para la leyenda
    gp2 << "NaN title 'Neuronas positivas' with points ps 2 pt 6 lw 3 lt rgb 'cyan', "
        << "NaN title 'Neuronas negativas' with points ps 2 pt 6 lw 3 lt rgb 'red'" << endl;

    getchar();

    return 0;
}
