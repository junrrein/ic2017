#include <armadillo>

using namespace std;
using namespace arma;

namespace ic {
// EstructruraCapaRed especifica la cantidad de neuronas en cada capa.
// Por ejemplo, si la red tiene 2 neuronas en la primera capa,
// 3 en la capa oculta y 1 en la capa de salida,
// EstructuraCapasRed para esa red será [2 3 1].
using EstructuraCapasRed = vec;

vec sigmoid(const vec& v, double b)
{
    return 2 / (1 + exp(-b * v)) - 1;
}

vec pendorcho(const vec& v)
{
    if (v.n_elem == 1)
        // Si la red tiene una sola salida, aplicar la función signo
        return v(0) >= 0 ? vec{1} : vec{-1};
    else {
        vec result = v;
        // Si la red tiene varias salidas, asignarle +1 a "la que ganó"
        // y -1 a las demás.
        const double maximo = max(result);

        result.transform([maximo](double val) {
            return val == maximo ? 1 : -1;
        });

        return result;
    }
}

double errorPrueba(const vector<mat>& pesos,
                   const mat& patrones,
                   const mat& salidaDeseada,
                   double parametroSigmoidea)
{
    int errores = 0;

    for (unsigned int n = 0; n < patrones.n_rows; ++n) {
        vector<vec> ySalidas;

        // Calculo de la salida para la primer capa
        {
            const vec v = pesos[0] * join_horiz(vec{-1}, patrones.row(n)).t(); // agrega entrada correspondiente al sesgo
            ySalidas.push_back(sigmoid(v, parametroSigmoidea));
        }

        // Calculo de las salidas para las demas capas
        // El número de capas es igual al número de matrices de pesos (una por capa)
        const int nCapas = pesos.size();
        for (int i = 1; i < nCapas; ++i) {
            const vec v = pesos[i] * join_vert(vec{-1}, ySalidas[i - 1]); // agrega entrada correspondiente al sesgo
            ySalidas.push_back(sigmoid(v, parametroSigmoidea));                            // TODO: ver qué valor le pasamos como parámetro
                                                                          // a la sigmoidea
        }

        const vec salidaRed = pendorcho(ySalidas.back()); // fija los valores en -1 o 1.

        if (any(salidaRed != salidaDeseada.row(n)))
            ++errores;
    }

    double tasaError = static_cast<double>(errores) / patrones.n_rows * 100;

    return tasaError;
}

pair<vector<mat>, double> epocaMulticapa(const mat& patrones,
                                         const mat& salidaDeseada,
                                         double tasaAprendizaje,
                                         double parametroSigmoidea,
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
            const vec v = nuevosPesos[0] * join_horiz(vec{-1}, patrones.row(n)).t(); // agrega entrada correspondiente al sesgo
            ySalidas.push_back(sigmoid(v, parametroSigmoidea));
        }

        // Calculo de las salidas para las demas capas
        for (int i = 1; i < nCapas; ++i) {
            const vec v = nuevosPesos[i] * join_vert(vec{-1}, ySalidas[i - 1]); // agrega entrada correspondiente al sesgo
            ySalidas.push_back(sigmoid(v, parametroSigmoidea));                 // TODO: ver qué valor le pasamos como parámetro
                                                                                // a la sigmoidea
        }

        // Calculo del error
        // Se quita tambien la componente correspondiente al sesgo
        const vec error = salidaDeseada.row(n).t() - ySalidas.back();

        // Calculo retropropagacion
        // Calculo de gradiente error local instantaneo
        vector<vec> delta;
        delta.resize(ySalidas.size());

        // Delta de ultima capa
        delta[delta.size() - 1] = error
                                  % (1 + ySalidas.back())
                                  % (1 - ySalidas.back());
        // Deltas de las capas anteriores
        for (int i = ySalidas.size() - 2; i >= 0; --i) {
            // No participan los pesos correspondientes al sesgo en el cálculo de los deltas
            const mat pesosAux = nuevosPesos[i + 1].tail_cols(nuevosPesos[i + 1].n_cols - 1);
            delta[i] = (pesosAux.t() * delta[i + 1])
                       % (1 + ySalidas[i])
                       % (1 - ySalidas[i]);
        }

        // Actualizacion de pesos de todas las capas menos la primera
        for (int i = nuevosPesos.size() - 1; i >= 1; --i) {
            const mat deltaW = tasaAprendizaje
                               * delta[i]
                               * join_horiz(vec{-1}, ySalidas[i - 1].t());
            nuevosPesos[i] += deltaW;
        }
        // Actualización de pesos de la primer capa
        const mat deltaW = tasaAprendizaje
                           * delta[0]
                           * join_horiz(vec{-1}, patrones.row(n));
        nuevosPesos[0] += deltaW;
    }

    // Calculo de Tasa de error
    double tasaError = errorPrueba(nuevosPesos, patrones, salidaDeseada, parametroSigmoidea);

    return {nuevosPesos, tasaError};
} // fin funcion Epoca

pair<vector<mat>, double> entrenarMulticapa(const EstructuraCapasRed& estructura,
                                            const mat& datos,
                                            int nEpocas,
                                            double tasaAprendizaje,
                                            double parametroSigmoidea,
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
    // y tantas columnas como componentes tiene la entrada, más la entrada correspondiente
    // al sesgo.
    pesos.push_back(randu<mat>(estructura(0), nEntradas + 1) - 0.5);
    for (int i = 1; i < nCapas; ++i) {
        // Las siguientes matrices de pesos tienen tantas filas como neuronas en dicha capa
        // y tantas columnas como entradas a esa capa, que van a ser las salidas de
        // la capa anterior mas la entrada correspondiente al sesgo.
        // Las salidas de la capa anterior es igual al nro de neuronas en la capa anterior.
        pesos.push_back(randu<mat>(estructura(i), estructura(i - 1) + 1) - 0.5);
    }

    double tasaError = 100;

    // Ciclo de las epocas
    for (int epoca = 1; epoca <= nEpocas; ++epoca) {
        // Ciclo para una época
        tie(pesos, tasaError) = epocaMulticapa(patrones,
                                               salidaDeseada,
                                               tasaAprendizaje,
                                               parametroSigmoidea,
                                               pesos);

        if (tasaError < toleranciaError)
            break;
    }
    // Fin ciclo (epocas)

    return {pesos, tasaError};
}

struct ParametrosMulticapa {
    EstructuraCapasRed estructuraRed;
    int nEpocas;
    double tasaAprendizaje;
    double parametroSigmoidea;
    double toleranciaError;
};
}

istream& operator>>(istream& is, ic::EstructuraCapasRed& estructura)
{
    // Formato estructura:
    // [2 3 1]
    {
    char ch = ' ';
    is >> ch;
    if (ch != '['){
        is.clear(ios::failbit);
        return is;
    }
    }

    // Asegurarnos de que la variable 'estructura' esté vacía
    estructura.clear();

    // Primero chequear si estamos por leer un número (con el primer caracter)
    // y después leerlo posta.
    for (char ch; is >> ch;) {
        if (isdigit(ch)) {
            is.unget();
            int numero;
            is >> numero;

            if (numero == 0) {      // No podemos tener una capa con 0 neuronas
                is.clear(ios::failbit);
                return is;
            }

            estructura.insert_rows(estructura.n_elem, vec{static_cast<double>(numero)});
        }
        else if (ch == ']') {       // Cuando se encuentra el corchete que cierra,
            break;                  // se terminó de leer la estructura.
        }
        else {                      // Si lo que se leyó no es número ni corchete, la estructura leída
            is.clear(ios::failbit); // tiene formato erróneo.
            return is;
        }
    }

    if (estructura.empty()) // Fallar si lo que se leyó es "[]"
        is.clear(ios::failbit);

    return is;
}

istream& operator>>(istream& is, ic::ParametrosMulticapa& parametros)
{
    // Formato:
    // estructura: [3 2 1]
    // n_epocas: 200
    // tasa_entrenamiento: 0.1
    // parametro_sigmoidea: 1
    // tolerancia_error: 5
    string str;

    // No chequeamos si la etiqueta de cada línea está bien o no. No nos importa
    is >> str >> parametros.estructuraRed
       >> str >> parametros.nEpocas
       >> str >> parametros.tasaAprendizaje
       >> str >> parametros.parametroSigmoidea
       >> str >> parametros.toleranciaError;

    // Control básico de valores de parámetros
    if (parametros.nEpocas <= 0
        || parametros.tasaAprendizaje <= 0
        || parametros.parametroSigmoidea <= 0
        || parametros.toleranciaError <= 0 || parametros.toleranciaError >= 100)
        is.clear(ios::failbit);

    return is;
}
