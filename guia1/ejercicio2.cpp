#include <iostream>
#include <armadillo>
#include "../config.hpp"

using namespace std;
using namespace arma;

pair<vec, double> entrenarPerceptron(const mat& datos,
                                     int nEpocas,
                                     double tasaAprendizaje,
                                     double toleranciaError);
double errorPrueba(const vec& pesos,
                   const mat& patrones,
                   const vec& salidaDeseada);

using Particion = pair<mat, mat>;
vector<Particion> particionar(mat datos, int nParticiones, double porcentajeEnt);

int main()
{
    arma_rng::set_seed_random();
    mat datos;

    // Primera parte del ejercicio (spheres1d)
    {
        datos.load(config::sourceDir + "/guia1/icgtp1datos/spheres1d10.csv");

        // FIXME:
        // 1: Las particiones tendrían que contener los índices a usar, no replicar los datos.
        // 2: Las particiones tendrían que ser generadas una sola vez, guardadas
        //    en un archivo, y luego levantarlas para que de esta manera poder repetir
        //    sesiones de entrenamiento/prueba con distintos clasificadores en
        //    los mismos datos. Incluso fijar la semilla de generación de núeros aleatorios.
        const vector<Particion> particiones = particionar(datos, 5, 80);

        vec errores;
        errores.set_size(particiones.size());
        int i = 0;

        for (const Particion& particion : particiones) {
            vec pesos;
            tie(pesos, ignore) = entrenarPerceptron(particion.first, 100, 0.1, 20);

            const double tasaError = errorPrueba(pesos,
                                                 particion.second.head_cols(3),
                                                 particion.second.tail_cols(1));
            errores[i++] = tasaError;
        }

        cout << "La validación cruzada del perceptron en spheres1d10 da como error:\n"
             << "Media: " << mean(errores) << '\n'
             << "Varianza: " << var(errores) << endl;
    }

    // Segunda parte del ejercicio (spheres2d)
    vector<string> archivos = {config::sourceDir + "/guia1/icgtp1datos/spheres2d10.csv",
                               config::sourceDir + "/guia1/icgtp1datos/spheres2d50.csv",
                               config::sourceDir + "/guia1/icgtp1datos/spheres2d70.csv"};

    for (string archivo : archivos) {
        datos.load(archivo);
        const vector<Particion> particiones = particionar(datos, 10, 80);

        vec errores;
        errores.set_size(particiones.size());
        int i = 0;

        for (const Particion& particion : particiones) {
            vec pesos;
            tie(pesos, ignore) = entrenarPerceptron(particion.first, 100, 0.1, 1);

            const double tasaError = errorPrueba(pesos,
                                                 particion.second.head_cols(3),
                                                 particion.second.tail_cols(1));
            errores[i++] = tasaError;
        }

        // FIXME:
        // Para que los resultados sean más informativos habría que proporcionar tambíen lo siguiente:
        // (para cada partición)
        // - Epocas que demoró en converger
        // - Error
        // Esto sirve para detectar (por ejemplo):
        // - Si siempre está terminando el entrenamiento por límite de épocas
        // - Si los errores están dando siempre altos
        //      Esto puede querer decir que la tolerancia de error que le estamos pidiendo es
        //      demasiado alta.
        cout << "La validación cruzada del perceptron en " << archivo << " da como error:\n"
             << "Media: " << mean(errores) << '\n'
             << "Varianza: " << var(errores) << endl;
    }

    return 0;
}

// Lo siguiente es validación cruzada clásica
vector<Particion> particionar(mat datos, int nParticiones, double porcentajeEnt)
{
    vector<Particion> particiones;
    const int nPatronesEnt = datos.n_rows * porcentajeEnt / 100;

    for (int i = 0; i < nParticiones; ++i) {
        datos = shuffle(datos); // Mezcla las filas de los datos
        particiones.push_back({datos.head_rows(nPatronesEnt),
                               datos.tail_rows(datos.n_rows - nPatronesEnt)});
    }

    return particiones;
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

pair<vec, double> entrenarPerceptron(const mat& datos,
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
    for (int epoca = 1; epoca <= nEpocas; ++epoca) {
        // Ciclo para una época
        tie(pesos, tasaError) = epocaPerceptron(patronesExt,
                                                salidaDeseada,
                                                tasaAprendizaje,
                                                pesos);

        if (tasaError < toleranciaError)
            break;
    }
    // Fin ciclo (epocas)

    return {pesos, tasaError};
}
