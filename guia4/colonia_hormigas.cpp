#include <armadillo>
#include <vector>
#include <set>
#include <map>

using namespace std;
using namespace arma;

struct Hormiga {
    vector<int> camino;

    double costoCamino(const mat& matrizCostos);
};

double Hormiga::costoCamino(const mat& distancias)
{
    double costo = 0;

    for (unsigned int i = 0; i < camino.size() - 1; ++i)
        costo += distancias.at(camino.at(i), camino.at(i + 1));

    return costo;
}

class ColoniaHormigas {
public:
    ColoniaHormigas(string rutaArchivoDistancias,
                    int nHormigas,
                    double sigma_cero,
                    double alpha,
                    double beta,
                    double Q);

    Hormiga buscarCamino();
    int seleccionarVecino(int ciudadActual, set<int> vecinos);

private:
    vector<Hormiga> m_hormiguero;
    mat m_distancias;
    mat m_feromonas;
    const int m_nHormigas;
    const double m_alpha;
    const double m_beta;
    const double m_Q;
    int m_nCiudades;
    int m_nodoOrigen;
};

ColoniaHormigas::ColoniaHormigas(string rutaArchivoDistancias,
                                 int nHormigas,
                                 double sigma_cero,
                                 double alpha,
                                 double beta,
                                 double Q)
    : m_nHormigas{nHormigas}
    , m_alpha{alpha}
    , m_beta{beta}
    , m_Q{Q}
{
    m_distancias.load(rutaArchivoDistancias);

    if (!m_distancias.is_square())
        throw runtime_error("La matriz leída no es cuadrada");

    m_nCiudades = m_distancias.n_rows;
    m_nodoOrigen = randi(1, distr_param(1, m_nCiudades)).at(0);
    m_feromonas = randu(m_distancias.n_rows, m_distancias.n_cols) * sigma_cero;
}

Hormiga ColoniaHormigas::buscarCamino()
{
    Hormiga hormiga;
    hormiga.camino = {m_nodoOrigen};

    set<int> vecinos;
    for (int i = 1; i <= m_nCiudades; ++i)
        vecinos.insert(i);

    vecinos.erase(m_nodoOrigen);

    // seleccionar nodo

    //meterlo en la lista
    // remover de vecinos

    return hormiga;
}

int ColoniaHormigas::seleccionarVecino(int ciudadActual,
                                       set<int> vecinos)
{
    map<int, double> probabilidadVecino;
    double sumaProbabilidades = 0;

    // Calculo el numerador y el denominador de la expresión de la diapositiva
    for (int vecino : vecinos) {
        const double probabilidad = pow(m_feromonas(ciudadActual - 1, vecino - 1), m_alpha)
                                    / pow(m_distancias(ciudadActual - 1, vecino - 1), m_beta);
        probabilidadVecino[vecino] = probabilidad;
        sumaProbabilidades += probabilidad;
    }

    // Normalizo cada numerador por el denominador calculado
    for (auto& p : probabilidadVecino)
        p.second /= sumaProbabilidades;

    // Ahora que tengo las probabilidades normalizadas,
    // selecciono un vecino
    const double moneda = randu(1).at(0, 0);
    double probAcumulada = 0;

    for (const auto& p : probabilidadVecino) {
        probAcumulada += p.second;

        if (moneda <= probAcumulada)
            return p.first;
    }

    throw runtime_error("Nunca se debería llegar hasta acá");
}
