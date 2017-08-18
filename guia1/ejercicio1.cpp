// http://arma.sourceforge.net/docs.html#example_prog

#include <iostream>
#include <armadillo>
#include <gnuplot-iostream.h>

using namespace std;
using namespace arma;

pair<vec, double> entrenarPerceptron(const mat& patronesEnt,
                                     const mat& patronesPrueba,
                                     const vec& salidaDeseadaEnt,
                                     const vec& salidaDeseadaPrueba,
                                     int nEpocas,
                                     double tasaAprendizaje,
                                     double tolerancia);

int main()
{
    arma_rng::set_seed_random();

    mat datos;
    datos.load("OR_trn.csv");
    mat patronesEntOR = datos.head_cols(2);
    vec salidaDeseadaEntOR = datos.tail_cols(1);
    datos.load("OR_tst.csv");
    mat patronesPruebaOR = datos.head_cols(2);
    vec salidaDeseadaPruebaOR = datos.tail_cols(1);

    // OR
    // Entrenar la red graficando resultados intermedios (la recta)
    vec pesos;
    double tasaError;
    tie(pesos, tasaError) = entrenarPerceptron(patronesEntOR,
                                               patronesPruebaOR,
                                               salidaDeseadaEntOR,
                                               salidaDeseadaPruebaOR,
                                               100,
                                               0.9,
                                               90);
    cout << "tasa de error del OR : " << tasaError << endl;

    // XOR
    // Entrenar la red graficando resultados intermedios (la recta)
    // Prueba

    //    mat patrones = ORTraining.head_cols(2);

    //    Gnuplot gp;
    //    gp << "plot " << gp.file1d(patrones) << "with points" << endl;

    //    getchar();

    return 0;
}

namespace ic {
int sign(double numero)
{
    if (numero >= 0)
        return 1;
    else
        return -1;
}
}

pair<vec, double> entrenarPerceptron(const mat& patronesEnt,
                                     const mat& patronesPrueba,
                                     const vec& salidaDeseadaEnt,
                                     const vec& salidaDeseadaPrueba,
                                     int nEpocas,
                                     double tasaAprendizaje,
                                     double tolerancia)
{
    // Inicializar pesos y tasa de error
    vec pesos = randu<vec>(patronesEnt.n_cols + 1) - 0.5;
    double tasaError = 0;

    // Extender la matriz de patrones con la entrada correspondiente al umbral
    mat patronesExt = join_horiz(ones(patronesEnt.n_rows) * (-1), patronesEnt);

    // Separar patrones en los casos verdaderos y falsos
    mat verdaderos;
    mat falsos;
    for (unsigned int i = 0; i < salidaDeseadaEnt.n_elem; ++i) {
        if (salidaDeseadaEnt(i) == 1)
            verdaderos.insert_rows(verdaderos.n_rows, patronesEnt.row(i));
        else
            falsos.insert_rows(falsos.n_rows, patronesEnt.row(i));
    }

    // Ciclo de las epocas
    for (int epoca = 1; epoca <= nEpocas; ++epoca) {
        const char caracterSalteo = 's';
        char caracter = ' ';
        // Ciclo para una época
        Gnuplot gp;

        for (unsigned int i = 0; i < patronesEnt.n_rows; ++i) {
            double z = dot(patronesExt.row(i), pesos);
            int y = ic::sign(z);

            // Actualizar pesos
            pesos += tasaAprendizaje * (salidaDeseadaEnt(i) - y) * patronesExt.row(i).t();

            // Graficar recta
            double pendiente = -pesos(1) / pesos(2);
            double ordenadaOrigen = pesos(0) / pesos(2);
            mat puntosRecta = {{-2, -2 * pendiente + ordenadaOrigen},
                               {2, 2 * pendiente + ordenadaOrigen}};

            if (caracter != caracterSalteo) {
                gp << "set grid\n"
                   << "unset key\n"
                   << "plot " << gp.file1d(verdaderos) << "with points, "
                   << gp.file1d(falsos) << "with points, "
                   << gp.file1d(puntosRecta) << "with lines" << endl;

                caracter = getchar();
            }
        }
        // Fin ciclo (época)

        int errores = 0;
        // Extender la matriz de patrones con la entrada correspondiente al umbral
        patronesExt = join_horiz(ones(patronesPrueba.n_rows) * (-1), patronesPrueba);

        for (unsigned int i = 0; i < patronesPrueba.n_rows; ++i) {
            double z = dot(patronesExt.row(i), pesos);
            int y = ic::sign(z);

            if (y != salidaDeseadaPrueba(i))
                ++errores;
        }

        double tasaError = errores / patronesPrueba.n_rows * 100;
        if (tasaError < tolerancia)
            break;
    }
    // Fin ciclo (epocas)

    return make_pair(pesos, tasaError);
}
