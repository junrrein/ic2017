// el perceptron simple se comporta, en promedio, peor para los datos del archivo
// spheres2d10 que para los datos de los archivos siguientes.
// Esto va en contra de nuestra intuicion porque esperamos que el clasificador se
// comporte mejor en datos con menor dispersion (mayor separacion entre las clases).
// Esto sucede porque al tener las clases mas separadas, el clasificador tiene mucho mas
// margen para ubicar la recta que las separa. En consecuencia, esta recta podria quedar ubicada
// demasiado cerca de alguna/s de las clases. Esto ocasiona que en la etapa de prueba pueda
// haber patrones de esta clase limitrofe que quedan justo del otro lado de la recta.
// En conclusion pueden resultar malos clasificadores bajo estas condiciones. Esto podria mejorarse
// con otras tecnicas como SVM.
// En los otros dos casos esto no sucede porque al estar mas disperso los datos, el clasificador tiene
// menos margen para ubicar la recta. Esto hace que sea mas probable que la recta encontrada haga una
// mejor separacion entre las clases.
#include <iostream>
#include <armadillo>

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
        datos.load("spheres1d10.csv");
        const vector<Particion> particiones = particionar(datos, 5, 80);

        vec errores;
        errores.set_size(particiones.size());
        int i = 0;

        for (const Particion& particion : particiones) {
            vec pesos;
            tie(pesos, ignore) = entrenarPerceptron(particion.first, 100, 0.1, 5);

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
    vector<string> archivos = {"spheres2d10.csv", "spheres2d50.csv", "spheres2d70.csv"};

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
        datos = shuffle(datos);
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
