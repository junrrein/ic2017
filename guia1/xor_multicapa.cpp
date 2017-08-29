#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

// EstructruraCapaRed especifica la cantidad de neuronas en cada capa.
// Por ejemplo, si la red tiene 2 neuronas en la primera capa,
// 3 en la capa oculta y 1 en la capa de salida,
// EstructuraCapasRed para esa red será [2 3 1].
using EstructuraCapasRed = vec;

pair<vector<mat>, double> entrenarMulticapa(const EstructuraCapasRed& estructura,
                                            const mat& datos,
                                            int nEpocas,
                                            double tasaAprendizaje,
                                            double toleranciaError);
double errorPrueba(const vec& pesos,
                   const mat& patrones,
                   const vec& salidaDeseada);

int main()
{
    arma_rng::set_seed_random();
    mat datos;
    datos.load("XOR_trn.csv");

    return 0;
}

namespace ic {
vec sigmoid(const vec& v, double b)
{
    vec result(v.n_elem);

    for (unsigned int i = 0; i < v.n_elem; ++i)
        result(i) = 2 / (1 + exp(-b * v(i))) - 1;

    return result;
}
}

pair<vec, double> epocaMulticapa(const mat& patronesExt,
                                 const mat& salidaDeseada,
                                 double tasaAprendizaje,
                                 const vector<mat>& pesos)
{
    //Entrenamiento
    vector<mat> nuevosPesos = pesos;
    const int nCapas = nuevosPesos.size();

    for (unsigned int n = 0; n < patronesExt.n_rows; ++n) {
        // Calcular las salidas para cada capa
        // También vamos a meter la entrada a la red en la estructura de salidas.
        // Esto es necesario para el cálculo de ajuste de pesos.
        vector<vec> ySalidas;
        ySalidas.push_back(patronesExt.row(n));

        for (int i = 1; i <= nCapas; ++i) {
            vec v = pesos[i - 1] * ySalidas[i - 1]; // FIXME: Esto va a explotar luego de la
                                                    // primer capa, porque no le estamos metiendo el sesgo
            ySalidas[i] = ic::sigmoid(v, 1);        // TODO: ver qué valor le pasamos como parámetro
                                                    // a la sigmoidea
        }
    }

    // Validacion
    int errores = 0;

    for (unsigned int i = 0; i < patronesExt.n_rows; ++i) {
        double z = dot(patronesExt.row(i), pesos);
        //        int y = ic::sign(z);

        //        if (y != salidaDeseada(i))
        //            ++errores;
    }

    double tasaError = static_cast<double>(errores) / patronesExt.n_rows * 100;

    return {nuevosPesos, tasaError};
} // fin funcion Epoca

double errorPrueba(const vec& pesos,
                   const mat& patrones,
                   const vec& salidaDeseada)
{
    // Se extiende la matriz de patrones con la entrada correspondiente al umbral
    const mat patronesExt = join_horiz(ones(patrones.n_rows) * (-1), patrones);

    int errores = 0;

    for (unsigned int i = 0; i < patronesExt.n_rows; ++i) {
        double z = dot(patronesExt.row(i), pesos);
        //        int y = ic::sign(z);

        //        if (y != salidaDeseada(i))
        //            ++errores;
    }

    double tasaError = static_cast<double>(errores) / patronesExt.n_rows * 100;

    return tasaError;
}

pair<vector<mat>, double> entrenarMulticapa(const EstructuraCapasRed& estructura,
                                            const mat& datos,
                                            int nEpocas,
                                            double tasaAprendizaje,
                                            double toleranciaError)
{
    const int nSalidas = estructura(estructura.n_elem - 1);
    const int nEntradas = datos.n_cols - nSalidas;
    const int nCapas = estructura.n_elem;
    // Vamos a tener tantas columnas en salidaDeseada
    // como neuronas en la capa de salida
    const mat salidaDeseada = datos.tail_cols(nSalidas);
    // Extender la matriz de patrones con la entrada correspondiente al umbral
    const mat patronesExt = join_horiz(ones(datos.n_rows) * (-1), datos.head_cols(nEntradas));

    // Inicializar pesos y tasa de error
    vector<mat> pesos;

    // La primer matriz matriz de pesos tiene tantas filas como neuronas en la primer capa
    // y tantas columnas como entradas.
    pesos.push_back(randu<mat>(estructura(0), nEntradas));
    for (int i = 1; i < nCapas; ++i) {
        // Las siguientes matrices de pesos tienen tantas filas como neuronas en dicha capa
        // y tantas columnas como entradas a esa capa, que van a ser las salidas de
        // la capa anterior.
        // Las salidas de la capa anterior es igual al nro de neuronas en la capa anterior.
        pesos.push_back(randu<mat>(estructura(i), estructura(i - 1)));
    }

    double tasaError = 0;

    // Ciclo de las epocas
    for (int epoca = 1; epoca <= nEpocas; ++epoca) {
        // Ciclo para una época
        tie(pesos, tasaError) = epocaMulticapa(patronesExt,
                                               salidaDeseada,
                                               tasaAprendizaje,
                                               pesos);

        if (tasaError < toleranciaError)
            break;
    }
    // Fin ciclo (epocas)

    return {pesos, tasaError};
}
