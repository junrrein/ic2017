#include <armadillo>

using namespace std;
using namespace arma;

namespace enjambre {

class Individuo {
public:
    Individuo(const vector<pair<double, double>>& limites);

    vec getPosicion() const { return posicion; }
    void setPosicion(const vec& value);

    vec getVelocidad() const { return velocidad; }
    void setVelocidad(const vec& value);

    vec getMejorLocal() const { return mejorLocal; }
    void setMejorLocal(const vec& value);

    double aptitudMejorLocal;

    Individuo& operator=(const Individuo& other);

private:
    const vector<pair<double, double>> m_limites;
    vec posicion;
    vec velocidad;
    vec mejorLocal;
};

Individuo::Individuo(const vector<pair<double, double>>& limites)
    : aptitudMejorLocal{-numeric_limits<double>::max()}
    , m_limites{limites}
{
    const unsigned int nVariables = m_limites.size();

    posicion = randu(nVariables);

    if (m_limites.size() != nVariables)
        throw runtime_error("Número de límites no concuerda con la cantidad de variables");

    for (unsigned int i = 0; i < posicion.n_elem; ++i) {
        double rangoVariable = m_limites.at(i).second - m_limites.at(i).first;

        posicion(i) = posicion(i) * rangoVariable + m_limites.at(i).first;
    }

    velocidad = zeros(nVariables);
    mejorLocal = posicion;
}

void Individuo::setPosicion(const vec& value)
{
    if (value.n_elem != this->posicion.n_elem)
        throw runtime_error("La dimensión del vector es incorrecta");

    for (unsigned int i = 0; i < m_limites.size(); ++i) {
        if (value.at(i) > m_limites.at(i).second)
            posicion.at(i) = m_limites.at(i).second; // Nos pasamos de largo por derecha
        else if (value.at(i) < m_limites.at(i).first)
            posicion.at(i) = m_limites.at(i).first; // Nos pasamos de largo por izquierda
        else
            posicion.at(i) = value.at(i); // No nos pasamos de largo
    }
}

void Individuo::setVelocidad(const vec& value)
{
    if (value.n_elem != this->velocidad.n_elem)
        throw runtime_error("La dimensión del vector es incorrecta");

    velocidad = value;
}

void Individuo::setMejorLocal(const vec& value)
{
    if (value.n_elem != this->posicion.n_elem)
        throw runtime_error("La dimensión del vector es incorrecta");

    mejorLocal = value;
}

Individuo& Individuo::operator=(const Individuo& other)
{
    if (other.m_limites != this->m_limites)
        throw runtime_error("Los individuos tienen distintos limites en las variables");

    this->posicion = other.posicion;
    this->velocidad = other.velocidad;
    this->mejorLocal = other.mejorLocal;
    this->aptitudMejorLocal = other.aptitudMejorLocal;

    return *this;
}

class Enjambre {
public:
    Enjambre(function<double(Individuo)> fitness,
             const vector<pair<double, double>>& limites,
             const double c1,
             const double c2,
             unsigned int nIndividuos,
             int umbral);

    bool evaluarPoblacion();
    void epoca();
    const vector<Individuo>& individuos() const { return m_individuos; }
    const Individuo& mejorGlobal() const { return m_mejorGlobal; }
    double fitnessPromedio() const { return m_fitnessPromedio; }
    bool termino() const { return m_termino; }

private:
    vector<Individuo> m_individuos;
    Individuo m_mejorGlobal;
    double m_mejorAptitud;
    double m_fitnessPromedio;
    function<double(Individuo)> m_fitness;
    const double m_c1, m_c2;
    int m_umbral;
    int m_epocasSinMejora;
    bool m_termino;
};

Enjambre::Enjambre(function<double(Individuo)> fitness,
                   const vector<pair<double, double>>& limites,
                   const double c1,
                   const double c2,
                   unsigned int nIndividuos,
                   int umbral)
    : m_mejorGlobal{limites}
    , m_mejorAptitud{-numeric_limits<double>::max()}
    , m_fitness{fitness}
    , m_c1{c1}
    , m_c2{c2}
    , m_umbral{umbral}
    , m_epocasSinMejora{0}
    , m_termino{false}
{
    for (unsigned int i = 0; i < nIndividuos; ++i) {
        m_individuos.push_back(Individuo{limites});

        evaluarPoblacion();
    }
}

bool Enjambre::evaluarPoblacion()
{
    bool mejoro = false;
    double sumaFitness = 0;

    for (Individuo& ind : m_individuos) {
        double aptitud = m_fitness(ind);
        sumaFitness += aptitud;

        if (aptitud > ind.aptitudMejorLocal) {
            ind.setMejorLocal(ind.getPosicion());
            ind.aptitudMejorLocal = aptitud;

            if (aptitud > m_mejorAptitud) {
                m_mejorGlobal = ind;
                m_mejorAptitud = aptitud;
                mejoro = true;
            }
        }
    }

    m_fitnessPromedio = sumaFitness / m_individuos.size();

    return mejoro;
}

void Enjambre::epoca()
{
    const int nVariables = m_individuos.front().getPosicion().n_elem;

    for (Individuo& ind : m_individuos) {
        // Cálculo de la nueva velocidad
        // Componente cognitiva
        const vec r1 = randu(nVariables);
        const vec distanciaMejorLocal = ind.getMejorLocal() - ind.getPosicion();

        const vec compCognitiva = m_c1 * r1 % distanciaMejorLocal;

        // Componente social
        const vec r2 = randu(nVariables);
        const vec distanciaMejorGlobal = m_mejorGlobal.getPosicion() - ind.getPosicion();

        const vec compSocial = m_c2 * r2 % distanciaMejorGlobal;

        // Actualización
        ind.setVelocidad(ind.getVelocidad() + compCognitiva + compSocial);
        ind.setPosicion(ind.getPosicion() + ind.getVelocidad());
    }

    if (evaluarPoblacion())
        m_epocasSinMejora = 0;
    else {
        if (m_epocasSinMejora == m_umbral)
            m_termino = true;

        ++m_epocasSinMejora;
    }
}
} // namespace enjambre
