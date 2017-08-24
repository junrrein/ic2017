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
                                     double porcentajeEnt,
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
            tie(pesos, ignore) = entrenarPerceptron(particion.first, 80, 1000, 0.05, 5);

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
            tie(pesos, ignore) = entrenarPerceptron(particion.first, 80, 1000, 0.1, 3);

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

pair<vec, double> epocaPerceptron(const mat& patronesEntExt,
                                  const mat& patronesValidacionExt,
                                  const vec& salidaDeseadaEnt,
                                  const vec& salidaDeseadaValidacion,
                                  double tasaAprendizaje,
                                  const vec& pesos)
{
    //Entrenamiento
    vec nuevosPesos = pesos;

    for (unsigned int i = 0; i < patronesEntExt.n_rows; ++i) {
        double z = dot(patronesEntExt.row(i), pesos);
        int y = ic::sign(z);

        // Actualizar pesos
        nuevosPesos += tasaAprendizaje * (salidaDeseadaEnt(i) - y) * patronesEntExt.row(i).t();
    } // Fin ciclo (Entrenamiento)

    // Validacion
    int errores = 0;

    for (unsigned int i = 0; i < patronesValidacionExt.n_rows; ++i) {
        double z = dot(patronesValidacionExt.row(i), pesos);
        int y = ic::sign(z);

        if (y != salidaDeseadaValidacion(i))
            ++errores;
    }

    double tasaError = static_cast<double>(errores) / patronesValidacionExt.n_rows * 100;

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
                                     double porcentajeEnt,
                                     int nEpocas,
                                     double tasaAprendizaje,
                                     double toleranciaError)
{
    const Particion p = particionar(datos, 1, porcentajeEnt).front();

    const vec salidaDeseadaEnt = p.first.tail_cols(1);
    const vec salidaDeseadaVal = p.second.tail_cols(1);
    // Extender la matriz de patrones con la entrada correspondiente al umbral
    const int nParametros = datos.n_cols - 1;
    const mat patronesEntExt = join_horiz(ones(p.first.n_rows) * (-1), p.first.head_cols(nParametros));
    const mat patronesValExt = join_horiz(ones(p.second.n_rows) * (-1), p.second.head_cols(nParametros));

    // Inicializar pesos y tasa de error
    vec pesos = randu<vec>(patronesEntExt.n_cols) - 0.5;
    vec mejoresPesos = pesos;
    vec ultimosErrores = {};
    double tasaErrorActual = 0;
    double mejorTasaError = 100;

    // Ciclo de las epocas
    for (int epoca = 1; epoca <= nEpocas; ++epoca) {
        // Ciclo para una época

        tie(pesos, tasaErrorActual) = epocaPerceptron(patronesEntExt,
                                                      patronesValExt,
                                                      salidaDeseadaEnt,
                                                      salidaDeseadaVal,
                                                      tasaAprendizaje,
                                                      pesos);

        if (tasaErrorActual < toleranciaError)
            break;

        // Guardar el error actual en el vector de ultimosErrores
        if (ultimosErrores.n_elem < 5)
            ultimosErrores.insert_rows(ultimosErrores.n_elem, tasaErrorActual);
        else {
            ultimosErrores = shift(ultimosErrores, -1);
            ultimosErrores(4) = tasaErrorActual;
        }

        // Verificar que los errores no vengan en aumento
        if (all(ultimosErrores > mejorTasaError))
            break;

        // Actualizar mejorError y mejoresPesos
        if (tasaErrorActual < mejorTasaError) {
            mejorTasaError = tasaErrorActual;
            mejoresPesos = pesos;
        }
    }
    // Fin ciclo (epocas)

    return {mejoresPesos, mejorTasaError};
}
