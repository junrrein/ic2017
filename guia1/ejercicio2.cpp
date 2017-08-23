// http://arma.sourceforge.net/docs.html#example_prog

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
            tie(pesos, std::ignore) = entrenarPerceptron(particion.first, 80, 100, 0.1, 5);

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
            tie(pesos, std::ignore) = entrenarPerceptron(particion.first, 80, 100, 0.1, 1);

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
    double tasaError = 0;
    bool isDescending = true;

    // Ciclo de las epocas
    for (int epoca = 1; epoca <= nEpocas; ++epoca) {
        // Ciclo para una época
        double tasaErrorAnterior = tasaError;

        tie(pesos, tasaError) = epocaPerceptron(patronesEntExt,
                                                patronesValExt,
                                                salidaDeseadaEnt,
                                                salidaDeseadaVal,
                                                tasaAprendizaje,
                                                pesos);

        if (tasaError < toleranciaError)
            break;

        // cortar si la tasa de error esta aumentando
        if (tasaError <= tasaErrorAnterior)
            isDescending = true;
        else {
            if (isDescending)
                isDescending = false; // la tasa de error empezo a aumentar
            else
                break; // la tasa de error venia en aumento
        }
    }
    // Fin ciclo (epocas)

    return {pesos, tasaError};
}
