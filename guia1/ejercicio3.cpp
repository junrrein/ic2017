#include "multicapa.cpp"
#include "particionar.cpp"
#include "../config.hpp"
#include <gnuplot-iostream.h>

mat reducirDimension(const mat& datos);
tuple<vec, double, int> entrenarPerceptron(const mat& datos,
                                           int nEpocas,
                                           double tasaAprendizaje,
                                           double toleranciaError);

int main()
{
    arma_rng::set_seed_random();

    mat datos;
    datos.load(config::sourceDir + "/guia1/icgtp1datos/concentlite.csv");
    //string rutaCarpeta = config::sourceDir + "/guia1/icgtp1datos/particionesConcent/";
    //const vector<ic::Particion> particiones = ic::cargarParticiones(rutaCarpeta, 10);

    // Parte a: Entrenar una red multicapa para clasificar los patrones

    ifstream ifs{config::sourceDir + "/guia1/parametrosConcentMulticapa.txt"};
    ic::ParametrosMulticapa parametros;
    if (!(ifs >> parametros))
        throw runtime_error{"Error al leer los parámetros"};

    vector<mat> pesos;
    double tasaError;
    int epoca;

    tie(pesos, tasaError, epoca) = ic::entrenarMulticapa(parametros.estructuraRed,
                                                         datos,
                                                         parametros.nEpocas,
                                                         parametros.tasaAprendizaje,
                                                         parametros.parametroSigmoidea,
                                                         parametros.toleranciaError);

    cout << "Multicapa sin inercia" << endl
         << "Tasa de error: " << tasaError << endl
         << "N° de epocas de entrenamiento: " << epoca << endl;

    mat falsosPositivos, falsosNegativos, verdaderosPositivos, verdaderosNegativos;

    for (unsigned int i = 0; i < datos.n_rows; i++) {
        const rowvec patron = datos.row(i).head(2);
        const double salidaDeseada = datos(i, 2);

        vec salidaRed = ic::salidaMulticapa(pesos, patron.t()).back();
        salidaRed = ic::pendorcho(salidaRed);

        if (salidaRed(0) == salidaDeseada) {
            if (salidaDeseada == 1)
                verdaderosPositivos.insert_rows(verdaderosPositivos.n_rows, patron);
            else
                verdaderosNegativos.insert_rows(verdaderosNegativos.n_rows, patron);
        }
        else {
            if (salidaDeseada == 1)
                falsosPositivos.insert_rows(falsosPositivos.n_rows, patron);
            else
                falsosNegativos.insert_rows(falsosNegativos.n_rows, patron);
        }
    }

    // Parte b : Experimentar qué pasa al usar el término de inercia
    // en el algoritmo de ajuste de pesos

    ifs.close();
    ifs.open(config::sourceDir + "/guia1/parametrosConcentInercia.txt");
    if (!(ifs >> parametros))
        throw runtime_error{"Error al leer los parámetros"};

    tie(ignore, tasaError, epoca) = ic::entrenarMulticapa(parametros.estructuraRed,
                                                          datos,
                                                          parametros.nEpocas,
                                                          parametros.tasaAprendizaje,
                                                          parametros.inercia,
                                                          parametros.parametroSigmoidea,
                                                          parametros.toleranciaError);

    cout << "\nMulticapa con inercia " << parametros.inercia << endl
         << "Tasa de error: " << tasaError << endl
         << "N° de epocas de entrenamiento: " << epoca << endl;

    // Parte c: Convertir los patrones a una sola dimensión, obtenida como
    // la distancia de cada patrón a la media total.
    // Luego entrenar con un perceptrón simple y comparar resultados con lo anterior.

    const mat datosReducidos = reducirDimension(datos);
    tie(ignore, tasaError, epoca) = entrenarPerceptron(datosReducidos, 500, 0.01, 4);

    cout << "\nPerceptrón simple en datos con dimensión reducida" << endl
         << "Tasa de error: " << tasaError << endl
         << "Epocas de entrenamiento: " << epoca << endl;

    // Graficar los resultados de clasificación de la primer red multicapa

    Gnuplot gp;

    gp << "set title 'Clasificación de patrones' font ',13'\n"
       << "set xlabel 'x_1' font ',11'\n"
       << "set ylabel 'x_2' font ',11'\n"
       << "set grid\n"
       << "set pointsize 2\n"
       << "plot " << gp.file1d(verdaderosPositivos) << "title 'Verdaderos Positivos' with points lt rgb 'blue', "
       << gp.file1d(verdaderosNegativos) << "title 'Verdaderos Negativos' with points lt rgb 'red', "
       << gp.file1d(falsosPositivos) << "title 'Falsos Positivos' with points lt rgb 'red' ps 1.5 pt 5, "
       << gp.file1d(falsosNegativos) << "title 'Falsos Negativos' with points lt rgb 'blue' ps 1.5 pt 5" << endl;

    getchar();

    return 0;
}

mat reducirDimension(const mat& datos)
{
    const mat patrones = datos.head_cols(2);
    const vec salidaDeseada = datos.tail_cols(1);
    const rowvec media = mean(patrones);
    vec distancias;
    distancias.resize(patrones.n_rows);

    for (unsigned int i = 0; i < patrones.n_rows; ++i) {
        const double distancia = norm(media - patrones.row(i));
        distancias(i) = distancia;
    }

    return join_horiz(distancias, salidaDeseada);
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

pair<vec, double> epocaPerceptron(const mat& patronesExt,
                                  const vec& salidaDeseada,
                                  double tasaAprendizaje,
                                  const vec& pesos)
{
    //Entrenamiento
    vec nuevosPesos = pesos;

    for (unsigned int i = 0; i < patronesExt.n_rows; ++i) {
        double z = dot(patronesExt.row(i), pesos);
        int y = ic::sign(z);

        // Actualizar pesos
        nuevosPesos += tasaAprendizaje * (salidaDeseada(i) - y) * patronesExt.row(i).t();
    } // Fin ciclo (Entrenamiento)

    // Validacion
    int errores = 0;

    for (unsigned int i = 0; i < patronesExt.n_rows; ++i) {
        double z = dot(patronesExt.row(i), pesos);
        int y = ic::sign(z);

        if (y != salidaDeseada(i))
            ++errores;
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
        int y = ic::sign(z);

        if (y != salidaDeseada(i))
            ++errores;
    }

    double tasaError = static_cast<double>(errores) / patronesExt.n_rows * 100;

    return tasaError;
}

tuple<vec, double, int> entrenarPerceptron(const mat& datos,
                                           int nEpocas,
                                           double tasaAprendizaje,
                                           double toleranciaError)
{
    const vec salidaDeseada = datos.tail_cols(1);
    // Extender la matriz de patrones con la entrada correspondiente al umbral
    const int nParametros = datos.n_cols - 1;
    const mat patronesExt = join_horiz(ones(datos.n_rows) * (-1), datos.head_cols(nParametros));

    // Inicializar pesos y tasa de error
    vec pesos = randu<vec>(patronesExt.n_cols) - 0.5;
    double tasaError = 0;

    // Ciclo de las epocas
    int epoca = 1;
    for (; epoca <= nEpocas; ++epoca) {
        // Ciclo para una época
        tie(pesos, tasaError) = epocaPerceptron(patronesExt,
                                                salidaDeseada,
                                                tasaAprendizaje,
                                                pesos);

        if (tasaError < toleranciaError)
            break;
    }
    // Fin ciclo (epocas)

    if (epoca > nEpocas)
        epoca = nEpocas;

    return make_tuple(pesos, tasaError, epoca);
}
