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
    return 2 / (1 + exp(-b * v)) - 1;
}

vec pendorcho(const vec& v)
{
    vec result = v;

    result.transform([](double val) {
        return val >= 0 ? 1 : -1;
    });

    return result;
}
}

pair<vector<mat>, double> epocaMulticapa(const mat& patrones,
                                         const mat& salidaDeseada,
                                         double tasaAprendizaje,
                                         const vector<mat>& pesos)
{
    //Entrenamiento
    vector<mat> nuevosPesos = pesos;
    const int nCapas = nuevosPesos.size();

    for (unsigned int n = 0; n < patrones.n_rows; ++n) {
        // Calcular las salidas para cada capa
        // También vamos a meter la entrada a la red en la estructura de salidas.
        // Esto es necesario para el cálculo de ajuste de pesos.
        vector<vec> ySalidas;

        // Calculo de la salida para la primer capa
        {
            const vec v = pesos[0] * join_horiz(vec{-1}, patrones.row(n)); // agrega entrada correspondiente al sesgo
            ySalidas.push_back(ic::sigmoid(v, 1));
        }

        // Calculo de las salidas para las demas capas
        for (int i = 1; i < nCapas; ++i) {
            const vec v = pesos[i] * join_vert(vec{-1}, ySalidas[i - 1]); // agrega entrada correspondiente al sesgo
            ySalidas.push_back(ic::sigmoid(v, 1));                        // TODO: ver qué valor le pasamos como parámetro
                                                                          // a la sigmoidea
        }

        // Calculo del error
        // Se quita tambien la componente correspondiente al sesgo
        const vec error = salidaDeseada - ySalidas.back();

        // Calculo retropropagacion
        // Calculo de gradiente error local instantaneo
        vector<vec> delta;
        delta.resize(ySalidas.size());
        // Delta de ultima capa
        delta[delta.size() - 1] = (0.5 * error % (1 + ySalidas.back()) % (1 - ySalidas.back()));

        // Deltas de las capas anteriores
        for (int i = ySalidas.size() - 2; i >= 0; --i) {
            delta[i] = (pesos[i + 1].t() * delta[i + 1]) * 0.5 % (1 + ySalidas[i]) % (1 - ySalidas[i]);
        }

        // Actualizacion de pesos
        for (int i = nuevosPesos.size() - 1; i >= 0; --i) {
            nuevosPesos[i] += tasaAprendizaje
                              * delta[i]
                              * join_horiz(vec{-1}, ySalidas[i - 1].t());
            // FIXME: Esto va a explotar al actualizar los pesos de la primer capa
        }
    }

    // Calculo de Tasa de error
    int errores = 0;

    for (unsigned int n = 0; n < patrones.n_rows; ++n) {
        vector<vec> ySalidas;

        // Calculo de la salida para la primer capa
        {
            const vec v = nuevosPesos[0] * join_horiz(vec{-1}, patrones.row(n)); // agrega entrada correspondiente al sesgo
            ySalidas.push_back(ic::sigmoid(v, 1));
        }

        // Calculo de las salidas para las demas capas
        for (int i = 1; i < nCapas; ++i) {
            const vec v = nuevosPesos[i] * join_vert(vec{-1}, ySalidas[i - 1]); // agrega entrada correspondiente al sesgo
            ySalidas.push_back(ic::sigmoid(v, 1));                              // TODO: ver qué valor le pasamos como parámetro
                                                                                // a la sigmoidea
        }

        const vec salidaRed = ic::pendorcho(ySalidas.back()); // fija los valores en -1 o 1.

        if (any(salidaRed != salidaDeseada.row(n)))
            ++errores;
    }

    double tasaError = static_cast<double>(errores) / patrones.n_rows * 100;

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
    const mat patrones = datos.head_cols(nEntradas);

    // Inicializar pesos y tasa de error
    vector<mat> pesos;

    // La primer matriz matriz de pesos tiene tantas filas como neuronas en la primer capa
    // y tantas columnas como entradas.
    pesos.push_back(randu<mat>(estructura(0), nEntradas + 1));
    for (int i = 1; i < nCapas; ++i) {
        // Las siguientes matrices de pesos tienen tantas filas como neuronas en dicha capa
        // y tantas columnas como entradas a esa capa, que van a ser las salidas de
        // la capa anterior mas la entrada correspondiente al sesgo.
        // Las salidas de la capa anterior es igual al nro de neuronas en la capa anterior.
        pesos.push_back(randu<mat>(estructura(i), estructura(i - 1) + 1));
    }

    double tasaError = 0;

    // Ciclo de las epocas
    for (int epoca = 1; epoca <= nEpocas; ++epoca) {
        // Ciclo para una época
        tie(pesos, tasaError) = epocaMulticapa(patrones,
                                               salidaDeseada,
                                               tasaAprendizaje,
                                               pesos);

        if (tasaError < toleranciaError)
            break;
    }
    // Fin ciclo (epocas)

    return {pesos, tasaError};
}
