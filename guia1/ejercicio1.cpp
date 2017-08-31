// http://arma.sourceforge.net/docs.html#example_prog

#include <iostream>
#include <armadillo>
#include <gnuplot-iostream.h>

using namespace std;
using namespace arma;

pair<vec, double> entrenarPerceptron(const mat& patrones,
                                     const vec& salidaDeseada,
                                     int nEpocas,
                                     double tasaAprendizaje,
                                     double tolerancia,
                                     string tituloGrafica);
double errorPrueba(const vec& pesos,
                   const mat& patrones,
                   const vec& salidaDeseada);

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
    tie(pesos, ignore) = entrenarPerceptron(patronesEntOR,
                                            salidaDeseadaEntOR,
                                            100,
                                            0.1,
                                            5,
                                            "OR");
    double tasaError = errorPrueba(pesos, patronesPruebaOR, salidaDeseadaPruebaOR);
    cout << "tasa de error del OR : " << tasaError << endl;

    // XOR
    // Entrenar la red graficando resultados intermedios (la recta)
    // Prueba
    datos.load("XOR_trn.csv");
    mat patronesEntXOR = datos.head_cols(2);
    vec salidaDeseadaEntXOR = datos.tail_cols(1);
    datos.load("XOR_tst.csv");
    mat patronesPruebaXOR = datos.head_cols(2);
    vec salidaDeseadaPruebaXOR = datos.tail_cols(1);

    tie(pesos, ignore) = entrenarPerceptron(patronesEntXOR,
                                            salidaDeseadaEntXOR,
                                            100,
                                            0.1,
                                            5,
                                            "XOR");
    tasaError = errorPrueba(pesos, patronesPruebaXOR, salidaDeseadaPruebaXOR);
    cout << "tasa de error del XOR : " << tasaError << endl;

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

pair<vec, double> entrenarPerceptron(const mat& patrones,
                                     const vec& salidaDeseada,
                                     int nEpocas,
                                     double tasaAprendizaje,
                                     double tolerancia,
                                     string tituloGrafica)
{
    // Inicializar pesos y tasa de error
    vec pesos = randu<vec>(patrones.n_cols + 1) - 0.5;
    double tasaError = 0;

    // Extender las matrices de patrones con la entrada correspondiente al umbral
    const mat patronesExt = join_horiz(ones(patrones.n_rows) * (-1), patrones);

    // Separar patrones de entrenamiento en casos verdaderos y falsos
    // para graficarlos en colores distintos
    const mat verdaderos = patrones.rows(find(salidaDeseada == 1));
    const mat falsos = patrones.rows(find(salidaDeseada == -1));

    // Objeto para graficar
    Gnuplot gp;
    // Caracter usado para dejar de graficar
    const char caracterSalteo = 's';
    char caracter = ' ';

    // Ciclo de las epocas
    for (int epoca = 1; epoca <= nEpocas; ++epoca) {
        // Ciclo para una época
        for (unsigned int i = 0; i < patrones.n_rows; ++i) {
            double z = dot(patronesExt.row(i), pesos);
            int y = ic::sign(z);

            if (caracter != caracterSalteo) {
                cout << "Patrón que entró: " << patrones.row(i)
                     << "Salida para ese patrón: " << y << '\n'
                     << "Salida deseada: " << salidaDeseada(i) << endl;

                caracter = getchar();
            }

            // Actualizar pesos
            // Tener en cuenta el orden en que colocamos la salida por la direccion del gradiente.
            pesos += tasaAprendizaje * 0.5 * (salidaDeseada(i) - y) * patronesExt.row(i).t();

            // Graficar recta
            double pendiente = -pesos(1) / pesos(2);
            double ordenadaOrigen = pesos(0) / pesos(2);
            mat puntosRecta = {{-2, -2 * pendiente + ordenadaOrigen},
                               {2, 2 * pendiente + ordenadaOrigen}};

            if (caracter != caracterSalteo) {
                gp << "set title '" << tituloGrafica << "' font ',13'\n"
                   << "set xlabel 'x_1'\n"
                   << "set ylabel 'x_2'\n"
                   << "set grid\n"
                   << "unset key\n"
                   << "plot " << gp.file1d(verdaderos) << "with points, "
                   << gp.file1d(falsos) << "with points, "
                   << gp.file1d(puntosRecta) << "with lines" << endl;
            }
        }
        // Fin ciclo (época)

        // Parte de prueba - Cálculo de la tasa de error
        int errores = 0;
        // Extender la matriz de patrones con la entrada correspondiente al umbral

        for (unsigned int i = 0; i < patronesExt.n_rows; ++i) {
            double z = dot(patronesExt.row(i), pesos);
            int y = ic::sign(z);

            if (y != salidaDeseada(i))
                ++errores;
        }

        tasaError = static_cast<double>(errores) / patrones.n_rows * 100;
        if (tasaError < tolerancia)
            break;
    }
    // Fin ciclo (epocas)

    return {pesos, tasaError};
}

double errorPrueba(const vec& pesos,
                   const mat& patrones,
                   const vec& salidaDeseada)
{
    // Se extiende la matriz de patrones con la entrada correspondiente al umbral
    const mat patronesExt = join_horiz(ones(patrones.n_rows) * (-1), patrones);

    int errores = 0;

    for (unsigned int i = 0; i < patronesExt.n_rows; ++i) {
        double z = dot(patronesExt.row(i), pesos);
        int y = ic::sign(z);

        if (y != salidaDeseada(i))
            ++errores;
    }

    double tasaError = static_cast<double>(errores) / patronesExt.n_rows * 100;

    return tasaError;
}
