#include <armadillo>
#include <vector>
#include <set>
#include <map>

using namespace std;
using namespace arma;

struct Hormiga {
    vector<int> camino;
    double costoCamino;
};

ostream& operator<<(ostream& os, const Hormiga& hormiga)
{
    os << "Recorrido: ";

    for (int ciudad : hormiga.camino)
        os << ciudad << ' ';

    os << "\nCosto del camino: " << hormiga.costoCamino << endl;

    return os;
}

class ColoniaHormigas {
public:
    ColoniaHormigas(string rutaArchivoDistancias,
                    int nHormigas,
                    int nEpocas,
                    double sigma_cero,
                    double alpha,
                    double beta,
                    double tasaEvaporacion,
                    double Q);

    double calcularCosto(const vector<int>& camino);
    Hormiga buscarCamino();
    int seleccionarVecino(int ciudadActual, set<int> vecinos);
    Hormiga encontrarSolucion();
    void depositarFeromonas();

private:
    vector<Hormiga> m_hormiguero;
    mat m_distancias;
    mat m_feromonas;
    const int m_nHormigas;
    const int m_nEpocas;
    const double m_alpha;
    const double m_beta;
    const double m_Q;
    const double m_tasaEvaporacion;
    unsigned int m_nCiudades;
    int m_nodoOrigen;
};

ColoniaHormigas::ColoniaHormigas(string rutaArchivoDistancias,
                                 int nHormigas,
                                 int nEpocas,
                                 double sigma_cero,
                                 double alpha,
                                 double beta,
                                 double tasaEvaporacion,
                                 double Q)
    : m_nHormigas{nHormigas}
    , m_nEpocas{nEpocas}
    , m_alpha{alpha}
    , m_beta{beta}
    , m_Q{Q}
    , m_tasaEvaporacion{tasaEvaporacion}
{
    m_distancias.load(rutaArchivoDistancias);

    if (!m_distancias.is_square())
        throw runtime_error("La matriz leída no es cuadrada");

    m_nCiudades = m_distancias.n_rows;
    m_nodoOrigen = randi(1, distr_param(1, m_nCiudades))(0);

    m_feromonas = randu(m_distancias.n_rows, m_distancias.n_cols) * sigma_cero;
    //    m_feromonas = mat(m_nCiudades, m_nCiudades);
    //    m_feromonas.fill(sigma_cero);
}

double ColoniaHormigas::calcularCosto(const vector<int>& camino)
{
    double costo = 0;

    for (unsigned int i = 0; i < camino.size() - 1; ++i) {
        const int ciudad1 = camino.at(i);
        const int ciudad2 = camino.at(i + 1);

        if (ciudad1 == ciudad2)
            throw runtime_error("Las ciudades son iguales");

        costo += m_distancias(ciudad1 - 1, ciudad2 - 1);
    }

    return costo;
}

Hormiga ColoniaHormigas::buscarCamino()
{
    Hormiga hormiga;
    hormiga.camino = {m_nodoOrigen};

    set<int> vecinos;
    for (unsigned int i = 1; i <= m_nCiudades; ++i)
        vecinos.insert(i);

    vecinos.erase(m_nodoOrigen);

    while (!vecinos.empty()) {
        const int ciudad = seleccionarVecino(hormiga.camino.back(), vecinos);
        hormiga.camino.push_back(ciudad);
        vecinos.erase(ciudad);
    }

    // Cuando salgo del bucle, pasé por todas las ciudades.
    // Solo falta volver a la ciudad inicial.
    hormiga.camino.push_back(m_nodoOrigen);

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
        //        const double probabilidad = pow(m_feromonas(ciudadActual - 1, vecino - 1), m_alpha);
        probabilidadVecino[vecino] = probabilidad;
        sumaProbabilidades += probabilidad;
    }

    // Normalizo cada numerador por el denominador calculado
    for (auto& p : probabilidadVecino)
        p.second /= sumaProbabilidades;

    // Ahora que tengo las probabilidades normalizadas,
    // selecciono un vecino
    const double moneda = randu(1).eval()(0);
    double probAcumulada = 0;

    for (const auto& p : probabilidadVecino) {
        probAcumulada += p.second;

        if (moneda <= probAcumulada)
            return p.first;
    }

    throw runtime_error("Nunca se debería llegar hasta acá");

    //    pair<int, double> mejor{0, 0};

    //    for (const auto& p : probabilidadVecino) {
    //        if (p.second > mejor.second)
    //            mejor = p;
    //    }

    //    return mejor.first;
}

Hormiga ColoniaHormigas::encontrarSolucion()
{
    for (int epoca = 1; epoca <= m_nEpocas; ++epoca) {
        m_hormiguero.clear();

        // Para cada hormiga, busco un camino que recorra todas
        // las ciudades.
        for (int i = 0; i < m_nHormigas; ++i)
            m_hormiguero.push_back(buscarCamino());

        // Me fijo si todas las hormigas tienen el mismo camino
        bool todosIguales = true;
        for (const Hormiga& hormiga : m_hormiguero) {
            if (hormiga.camino != m_hormiguero.front().camino) {
                todosIguales = false;
                break;
            }
        }

        if (todosIguales) {
            Hormiga solucion = m_hormiguero.front();
            solucion.costoCamino = calcularCosto(solucion.camino);

            return solucion;
        }

        // Evaporar feromonas
        m_feromonas *= (1 - m_tasaEvaporacion);

        // Depositar feromonas
        depositarFeromonas();
    }

    // Si se llega hasta acá las hormigas no convergieron
    // a un único camino.
    throw runtime_error("No se encontró una solución");
}

void ColoniaHormigas::depositarFeromonas()
{
    if (m_hormiguero.empty())
        throw runtime_error("El hormiguero está vacío");

    for (Hormiga& hormiga : m_hormiguero) {
        hormiga.costoCamino = calcularCosto(hormiga.camino);
        const double deltaFeromonas = m_Q / hormiga.costoCamino;

        for (unsigned int i = 0; i < hormiga.camino.size() - 1; ++i) {
            const int ciudad1 = hormiga.camino.at(i);
            const int ciudad2 = hormiga.camino.at(i + 1);

            if (ciudad1 == ciudad2)
                throw runtime_error("Las ciudades son iguales");

            m_feromonas(ciudad1 - 1, ciudad2 - 1) += deltaFeromonas;
            //			m_feromonas(ciudad2 - 1, ciudad1 - 1) += deltaFeromonas;
        }
    }
}
