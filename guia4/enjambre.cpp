#include <armadillo>

using namespace std;
using namespace arma;

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

private:
    vec posicion;
    vec velocidad;
    vec mejorLocal;
};

Individuo::Individuo(const vector<pair<double, double>>& limites)
    : aptitudMejorLocal{-numeric_limits<double>::max()}
{
    const unsigned int nVariables = limites.size();

    posicion = randu(nVariables);

    if (limites.size() != nVariables)
        throw runtime_error("Número de límites no concuerda con la cantidad de variables");

    for (unsigned int i = 0; i < posicion.n_elem; ++i) {
        double rangoVariable = limites.at(i).second - limites.at(i).first;

        posicion(i) = posicion(i) * rangoVariable + limites.at(i).first;
    }

    velocidad = zeros(nVariables);
    mejorLocal = posicion;
}

void Individuo::setPosicion(const vec& value)
{
    if (value.n_elem != this->posicion.n_elem)
        throw runtime_error("La dimensión del vector es incorrecta");

    posicion = value;
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

class Enjambre {
    Enjambre(function<double(Individuo)> fitness,
             const vector<pair<double, double>>& limites,
             unsigned int nIndividuos,
             int umbral);

    bool evaluarFitness();
    void epoca();
    const Individuo& mejorGlobal() const { return m_mejorGlobal; }

private:
    const unsigned int m_nIndividuos;
    vector<Individuo> m_individuos;
    Individuo m_mejorGlobal;
    double m_mejorAptitud;
    function<double(Individuo)> m_fitness;
    int m_umbral;
    int m_epocasSinMejora;
    bool m_termino;
};

Enjambre::Enjambre(function<double(Individuo)> fitness,
                   const vector<pair<double, double>>& limites,
                   unsigned int nIndividuos,
                   int umbral)
    : m_nIndividuos{nIndividuos}
    , m_mejorGlobal{limites}
    , m_fitness{fitness}
    , m_umbral{umbral}
    , m_epocasSinMejora{0}
    , m_termino{false}
{
    for (unsigned int i = 0; i < m_nIndividuos; ++i) {
        m_individuos.push_back(Individuo{limites});

        evaluarFitness();
    }
}

bool Enjambre::evaluarFitness()
{
    bool mejoro = false;

    for (Individuo& ind : m_individuos) {
        double aptitud = m_fitness(ind);

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

    return mejoro;
}

void Enjambre::epoca()
{
    const int nVariables = m_individuos.front().getPosicion().n_elem;

    for (Individuo& ind : m_individuos) {
        // Cálculo de la nueva velocidad
        // Componente cognitiva
        const double c1 = 1;
        const vec r1 = randu(nVariables);
        const vec distanciaMejorLocal = ind.getMejorLocal() - ind.getPosicion();

        const vec compCognitiva = c1 * r1 % distanciaMejorLocal;

        // Componente social
        const double c2 = 1;
        const vec r2 = randu(nVariables);
        const vec distanciaMejorGlobal = m_mejorGlobal.getPosicion() - ind.getPosicion();

        const vec compSocial = c2 * r2 % distanciaMejorGlobal;

        // Actualización
        ind.setVelocidad(ind.getVelocidad() + compCognitiva + compSocial);
        ind.setPosicion(ind.getPosicion() + ind.getVelocidad());
    }

    if (evaluarFitness())
        m_epocasSinMejora = 0;
    else {
        if (m_epocasSinMejora == m_umbral)
            m_termino = true;

        ++m_epocasSinMejora;
    }
}
