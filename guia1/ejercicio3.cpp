#include "multicapa.cpp"
#include "particionar.cpp"
#include "../config.hpp"
#include <gnuplot-iostream.h>

mat reducirDimension(const mat& datos);
tuple<vec, double, int> entrenarPerceptron(const mat& datos,
                                           int nEpocas,
                                           double tasaAprendizaje,
                                           double toleranciaError);
double errorPerceptron(const vec& pesos,
                       const mat& patrones,
                       const vec& salidaDeseada);
double errorPerceptron(const vec& pesos,
                       const mat& datos);

int main()
{
    arma_rng::set_seed_random();

    // Vamos a usar las mismas semillas para inicializar los pesos
    // en los distintos algoritmos de entrenamiento
    const ivec semillas = randi(10);

    mat datos;
    datos.load(config::sourceDir + "/guia1/icgtp1datos/concentlite.csv");
    string rutaCarpeta = config::sourceDir + "/guia1/icgtp1datos/particionesConcent/";
    const vector<ic::Particion> particiones = ic::cargarParticiones(rutaCarpeta, 10);

    // Parte a: Entrenar una red multicapa sin inercia para clasificar los patrones

    ifstream ifs{config::sourceDir + "/guia1/parametrosConcentMulticapa.txt"};
    ic::ParametrosMulticapa parametros;
    if (!(ifs >> parametros))
        throw runtime_error{"Error al leer los parámetros"};

    vec errores, epocas;
    errores.set_size(particiones.size());
    epocas.set_size(particiones.size());

#pragma omp parallel for
    for (unsigned int i = 0; i < particiones.size(); ++i) {
        vector<mat> pesos;
        int epoca;

        tie(pesos, ignore, ignore, epoca) = ic::entrenarMulticapa(parametros.estructuraRed,
                                                                  datos.rows(particiones[i].first),
                                                                  parametros.nEpocas,
                                                                  parametros.tasaAprendizaje,
                                                                  parametros.inercia,
                                                                  parametros.toleranciaError,
                                                                  semillas(i));

        double tasaError = ic::errorClasificacionMulticapa(pesos,
                                                           datos.rows(particiones[i].second));

        // Para cada partición, se guarda el número de épocas que tardó en converger
        // y la tasa de error de clasificación en la partición de prueba.
        epocas(i) = epoca;
        errores(i) = tasaError;
    }

    cout << "Multicapa sin inercia" << endl
         << "Tasa de error promedio en prueba: " << mean(errores) << endl
         << "Desvío estándar de la tasa de error: " << stddev(errores) << endl
         << "N° de épocas promedio que tarda en converger: " << mean(epocas) << endl
         << "Desvío estándar de lo anterior: " << stddev(epocas) << endl;

    // Parte b : Experimentar qué pasa al usar el término de inercia
    // en el algoritmo de ajuste de pesos

    ifs.close();
    ifs.open(config::sourceDir + "/guia1/parametrosConcentInercia.txt");
    if (!(ifs >> parametros))
        throw runtime_error{"Error al leer los parámetros"};

    errores.clear();
    epocas.clear();
    errores.set_size(particiones.size());
    epocas.set_size(particiones.size());

#pragma omp parallel for
    for (unsigned int i = 0; i < particiones.size(); ++i) {
        vector<mat> pesos;
        int epoca;

        tie(pesos, ignore, ignore, epoca) = ic::entrenarMulticapa(parametros.estructuraRed,
                                                                  datos.rows(particiones[i].first),
                                                                  parametros.nEpocas,
                                                                  parametros.tasaAprendizaje,
                                                                  parametros.inercia,
                                                                  parametros.toleranciaError,
                                                                  semillas(i));

        double tasaError = ic::errorClasificacionMulticapa(pesos,
                                                           datos.rows(particiones[i].second));

        epocas(i) = epoca;
        errores(i) = tasaError;
    }

    cout << "\nMulticapa con inercia " << parametros.inercia << endl
         << "Tasa de error promedio en prueba: " << mean(errores) << endl
         << "Desvío estándar de la tasa de error: " << stddev(errores) << endl
         << "N° de épocas promedio que tarda en converger: " << mean(epocas) << endl
         << "Desvío estándar de lo anterior: " << stddev(epocas) << endl;

    // Parte c: Convertir los patrones a una sola dimensión, obtenida como
    // la distancia de cada patrón a la media total.
    // Luego entrenar con un perceptrón simple y comparar resultados con lo anterior.

    const mat datosReducidos = reducirDimension(datos);

    errores.clear();
    epocas.clear();
    errores.set_size(particiones.size());
    epocas.set_size(particiones.size());

    for (unsigned int i = 0; i < particiones.size(); ++i) {
        vec pesos;
        int epoca;
        tie(pesos, ignore, epoca) = entrenarPerceptron(datosReducidos.rows(particiones[i].first),
                                                       1000,
                                                       0.01,
                                                       4);

        double tasaError = errorPerceptron(pesos,
                                           datosReducidos.rows(particiones[i].second));

        epocas(i) = epoca;
        errores(i) = tasaError;
    }

    cout << "\nPerceptrón simple en datos con dimensión reducida" << endl
         << "Tasa de error promedio en prueba: " << mean(errores) << endl
         << "Desvío estándar de la tasa de error: " << stddev(errores) << endl
         << "N° de épocas promedio que tarda en converger: " << mean(epocas) << endl
         << "Desvío estándar de lo anterior: " << stddev(epocas) << endl;

    // Ultima parte:
    // Entrenar un clasificador con inercia y graficar el resultado de la clasificación

    vector<mat> pesos;
    vec erroresClasificacion, erroresCuadraticos;
    int nEpocas;
    // Elegimos una semilla en particular para luego usarla para entrenar
    // un multicapa sin inercia y comparar velocidad de convergencia.
    const long long semilla = randi(1)(0);

    tie(pesos, erroresClasificacion, erroresCuadraticos, nEpocas) = ic::entrenarMulticapa(parametros.estructuraRed,
                                                                                          datos.rows(particiones[1].first),
                                                                                          parametros.nEpocas,
                                                                                          parametros.tasaAprendizaje,
                                                                                          parametros.inercia,
                                                                                          parametros.toleranciaError,
                                                                                          semilla);

    const mat datosPrueba = datos.rows(particiones[1].second);

    mat falsosPositivos, falsosNegativos, verdaderosPositivos, verdaderosNegativos;

    for (unsigned int i = 0; i < datosPrueba.n_rows; i++) {
        const rowvec patron = datosPrueba.row(i).head(2);
        const double salidaDeseada = datosPrueba(i, 2);

        vec salidaRed = ic::salidaMulticapa(pesos, patron.t()).back();
        salidaRed = ic::winnerTakesAll(salidaRed);

        if (salidaRed(0) == salidaDeseada) {
            if (salidaDeseada == 1)
                verdaderosPositivos.insert_rows(verdaderosPositivos.n_rows, patron);
            else
                verdaderosNegativos.insert_rows(verdaderosNegativos.n_rows, patron);
        }
        else {
            if (salidaDeseada == 1)
                falsosNegativos.insert_rows(falsosNegativos.n_rows, patron);
            else
                falsosPositivos.insert_rows(falsosPositivos.n_rows, patron);
        }
    }

    // Evaluar el resultado de la red en todo el plano.
    // A partir de esto se va a graficar la frontera de decisión
    // (los valores de y tal que y = 0)
    mat puntosSuperficie(50 * 50, 3);
    {
        const vec x = linspace(0, 1, 50);
        const vec y = linspace(0, 1, 50);

        for (unsigned int j = 0; j < x.n_elem; ++j) {
            for (unsigned int k = 0; k < y.n_elem; ++k) {
                const vector<vec> salidaRed = ic::salidaMulticapa(pesos, vec{x(j), y(k)});
                const double salidaFinal = salidaRed.back()(0);
                const rowvec punto = {x(j), y(k), salidaFinal};

                puntosSuperficie.row(50 * j + k) = punto;
            }
        }
    }

    // Graficar las fronteras de decisión de las neuronas de la primer capa en 2D
    // Primero calcular dichas fronteras de decisión
    vector<pair<double, double>> rectas;

    for (unsigned int i = 0; i < parametros.estructuraRed(0); ++i) {
        const double w0 = pesos[0](i, 0);
        const double w1 = pesos[0](i, 1);
        const double w2 = pesos[0](i, 2);
        const double pendiente = -w1 / w2;
        const double ordenada = w0 / w2;

        rectas.push_back({pendiente, ordenada});
    }

    // Función que devuelve el string que Gnuplot usa para graficar una recta.
    // Lo que devuelve tiene la forma "x * pendiente  + ordenda".
    auto stringRecta = [](double pendiente, double ordenada) {
        return "x * " + to_string(pendiente) + " + " + to_string(ordenada) + " ";
    };

    Gnuplot gp;

    gp << "set xrange [0:1]" << endl
       << "set yrange [0:1]" << endl
       << "set title 'Clasificación de patrones' font ',13'" << endl
       << "set xlabel 'x_1' font ',11'" << endl
       << "set ylabel 'x_2' font ',11'" << endl
       << "set pointsize 2" << endl
       << "set grid linewidth 1" << endl

       // Se plotea la frontera de decisión a un archivo
       << "set contour base" << endl
       << "set cntrparam levels discrete 0" << endl
       << "set dgrid3d 50, 50" << endl
       << "unset surface" << endl
       << "set table 'frontera.dat'" << endl
       << "splot " << gp.file1d(puntosSuperficie) << endl
       << "unset table" << endl
       << "unset dgrid3d" << endl
       << "unset contour" << endl

       // Ploteo final
       << "plot " << gp.file1d(verdaderosPositivos) << "title 'Verdaderos Positivos' with points lt rgb 'blue', "
       << gp.file1d(verdaderosNegativos) << "title 'Verdaderos Negativos' with points lt rgb 'red', "
       << gp.file1d(falsosPositivos) << "title 'Falsos Positivos' with points lt rgb 'red' ps 1.5 pt 5, "
       << gp.file1d(falsosNegativos) << "title 'Falsos Negativos' with points lt rgb 'blue' ps 1.5 pt 5, ";

    for (const auto& recta : rectas) {
        gp << stringRecta(recta.first, recta.second) << "notitle with lines lt rgb 'green', ";
    }

    // Agregar leyenda para las fronteras de decisión
    gp << "NaN title 'Fronteras de decisión intermedias' with lines lt rgb 'green', "
       << "'frontera.dat' title 'Frontera de decisión final' with lines lt -1 lw 1.5" << endl;

    getchar();

    // Graficar los resultados de clasificación de la red multicapa
    // Vamos a hacer un gráfico para cada neurona de la primer capa, mostrando
    // la superficie sigmoidea que genera esa neurona

    for (unsigned int i = 0; i < parametros.estructuraRed(0); ++i) {
        // Calcular los puntos de la superficie sigmoidea
        const vec x = linspace(0, 1, 20);
        const vec y = linspace(0, 1, 20);
        puntosSuperficie.clear();

        for (unsigned int j = 0; j < x.n_elem; ++j) {
            for (unsigned int k = 0; k < y.n_elem; ++k) {
                const vector<vec> salidaRed = ic::salidaMulticapa(pesos, vec{x(j), y(k)});
                const double salidaNeurona = salidaRed.front()(i);
                const rowvec punto = {x(j), y(k), salidaNeurona};

                puntosSuperficie.insert_rows(puntosSuperficie.n_rows, punto);
            }
        }

        gp << "set surface" << endl
           << "set zlabel 'y^1_" << to_string(i + 1) << "' font ',11'" << endl
           << "set grid linewidth 2" << endl
           << "set key box opaque" << endl
           << "set xyplane at 0" << endl
           << "set dgrid3d 20,20" << endl
           << "set hidden3d" << endl
           << "set table 'superficie.dat'" << endl
           << "splot " << gp.file1d(puntosSuperficie) << "using 1:2:3 with lines" << endl
           << "unset table" << endl
           << "unset dgrid3d" << endl
           << "splot " << gp.file1d(verdaderosPositivos) << "using 1:2:(0.0) title 'Verdaderos Positivos' with points lt rgb 'blue' pt 1, "
           << gp.file1d(verdaderosNegativos) << "using 1:2:(0.0) title 'Verdaderos Negativos' with points lt rgb 'red' pt 1, "
           << gp.file1d(falsosPositivos) << "using 1:2:(0.0) title 'Falsos Positivos' with points lt rgb 'red' ps 1.5 pt 5, "
           << gp.file1d(falsosNegativos) << "using 1:2:(0.0) title 'Falsos Negativos' with points lt rgb 'blue' ps 1.5 pt 5, "
           << "'superficie.dat' title 'Salida de la neurona " << to_string(i + 1) << "' with lines palette" << endl;

        getchar();
    }

    // Graficar la superficie que la red neuronal da como resultado
    {
        const vec x = linspace(0, 1, 20);
        const vec y = linspace(0, 1, 20);
        puntosSuperficie.clear();

        for (unsigned int j = 0; j < x.n_elem; ++j) {
            for (unsigned int k = 0; k < y.n_elem; ++k) {
                const vector<vec> salidaRed = ic::salidaMulticapa(pesos, vec{x(j), y(k)});
                const double salidaFinal = salidaRed.back()(0);
                const rowvec punto = {x(j), y(k), salidaFinal};

                puntosSuperficie.insert_rows(puntosSuperficie.n_rows, punto);
            }
        }

        gp << "set xrange [0:1]" << endl
           << "set yrange [0:1]" << endl
           << "set title 'Clasificación de patrones' font ',13'" << endl
           << "set xlabel 'x_1' font ',11'" << endl
           << "set ylabel 'x_2' font ',11'" << endl
           << "set zlabel 'y_{final}' font ',11'" << endl
           << "set grid linewidth 2" << endl
           << "set key box opaque" << endl
           << "set xyplane at 0" << endl
           << "set dgrid3d 20,20" << endl
           << "set hidden3d" << endl
           << "set table 'superficie.dat'" << endl
           << "splot " << gp.file1d(puntosSuperficie) << "using 1:2:3 with lines" << endl
           << "unset table" << endl
           << "unset dgrid3d" << endl
           << "splot " << gp.file1d(verdaderosPositivos) << "using 1:2:(0.0) title 'Verdaderos Positivos' with points lt rgb 'blue' pt 1, "
           << gp.file1d(verdaderosNegativos) << "using 1:2:(0.0) title 'Verdaderos Negativos' with points lt rgb 'red' pt 1, "
           << gp.file1d(falsosPositivos) << "using 1:2:(0.0) title 'Falsos Positivos' with points lt rgb 'red' ps 1.5 pt 5, "
           << gp.file1d(falsosNegativos) << "using 1:2:(0.0) title 'Falsos Negativos' with points lt rgb 'blue' ps 1.5 pt 5, "
           << "'superficie.dat' title 'Salida de la red' with lines palette" << endl;

        getchar();
    }

    // Graficar evolucion del error durante el entrenamiento
    gp << "set multiplot layout 2, 1 title 'Multicapa con inercia' font ',14'" << endl
       << "set xrange [1:" << nEpocas + 1 << "]" << endl
       << "set yrange [0:*]" << endl
       << "unset key" << endl
       << "set grid linewidth 1" << endl
       << "set title 'Evolución del error de clasificación'" << endl
       << "set xlabel 'Epoca'" << endl
       << "set ylabel 'Tasa de error (%)'" << endl
       << "plot " << gp.file1d(erroresClasificacion) << "with lines" << endl
       << "set title 'Evolución de la suma del error cuadrático'" << endl
       << "set ylabel 'Suma del error cuadrático'" << endl
       << "plot " << gp.file1d(erroresCuadraticos) << "with lines" << endl;

    // Mostrar la evolución del error en un multicapa sin inercia
    tie(pesos, erroresClasificacion, erroresCuadraticos, nEpocas) = ic::entrenarMulticapa(parametros.estructuraRed,
                                                                                          datos.rows(particiones[1].first),
                                                                                          parametros.nEpocas,
                                                                                          parametros.tasaAprendizaje,
                                                                                          0,
                                                                                          parametros.toleranciaError,
                                                                                          semilla);

    Gnuplot gp2;

    gp2 << "set multiplot layout 2, 1 title 'Multicapa sin inercia' font ',14'" << endl
        << "set xrange [1:" << nEpocas + 1 << "]" << endl
        << "set yrange [0:*]" << endl
        << "unset key" << endl
        << "set grid" << endl
        << "set title 'Evolución del error de clasificación' font ',13'" << endl
        << "set xlabel 'Epoca'" << endl
        << "set ylabel 'Tasa de error (%)'" << endl
        << "plot " << gp.file1d(erroresClasificacion) << "with lines" << endl
        << "set title 'Evolución de la suma del error cuadrático' font ',13'" << endl
        << "set ylabel 'Suma del error cuadrático'" << endl
        << "plot " << gp.file1d(erroresCuadraticos) << "with lines" << endl;

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

double errorPerceptron(const vec& pesos,
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

double errorPerceptron(const vec& pesos,
                       const mat& datos)
{
    return errorPerceptron(pesos,
                           datos.head_cols(datos.n_cols - 1),
                           datos.tail_cols(1));
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
