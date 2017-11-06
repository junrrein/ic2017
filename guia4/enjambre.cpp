#include <armadillo>

using namespace std;
using namespace arma;

class Individuo {
public:
    Individuo(unsigned int t_nVariables,
              const vector<pair<double, double>>& t_limites);

    vec getPosicion() const { return posicion; }
    void setPosicion(const vec& value);

    vec getVelocidad() const { return velocidad; }
    void setVelocidad(const vec& value);

    vec getMejorLocal() const { return mejorLocal; }
    void setMejorLocal(const vec& value);

    double aptitudMejorLocal;

    Individuo& operator=(const Individuo& other);

private:
    vec posicion;
    const vector<pair<double, double>> limites;
    vec velocidad;
    vec mejorLocal;

    const unsigned int nVariables;
};

Individuo::Individuo(unsigned int t_nVariables,
                     const vector<pair<double, double>>& t_limites)
    : aptitudMejorLocal{-numeric_limits<double>::max()}
    , limites{t_limites}
    , nVariables{t_nVariables}
{
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
    if (value.n_elem != nVariables)
        throw runtime_error("La dimensión del vector es incorrecta");

    posicion = value;
}

void Individuo::setVelocidad(const vec& value)
{
    if (value.n_elem != nVariables)
        throw runtime_error("La dimensión del vector es incorrecta");

    velocidad = value;
}

void Individuo::setMejorLocal(const vec& value)
{
    if (value.n_elem != nVariables)
        throw runtime_error("La dimensión del vector es incorrecta");

    mejorLocal = value;
}

Individuo& Individuo::operator=(const Individuo& other)
{
    if (this->nVariables != other.nVariables)
        throw runtime_error("Los individuos tienen distinta cantidad de variables");

    this->posicion = other.posicion;
    this->velocidad = other.velocidad;
    this->mejorLocal = other.mejorLocal;
    this->aptitudMejorLocal = other.aptitudMejorLocal;

    return *this;
}

class Enjambre {
    Enjambre(unsigned int nVariables,
             function<double(Individuo)> fitness,
             const vector<pair<double, double>>& limites,
             unsigned int nIndividuos);

    void evaluarFitness();

private:
    const unsigned int m_nIndividuos;
    vector<Individuo> m_individuos;
    Individuo m_mejorGlobal;
    double m_mejorAptitud;
    function<double(Individuo)> m_fitness;
    int m_generacionesSinMejora;
};

Enjambre::Enjambre(unsigned int nVariables,
                   function<double(Individuo)> fitness,
                   const vector<pair<double, double>>& limites,
                   unsigned int nIndividuos)
    : m_nIndividuos{nIndividuos}
    , m_mejorGlobal{nVariables, limites}
    , m_fitness{fitness}
    , m_generacionesSinMejora{0}
{
    for (unsigned int i = 0; i < m_nIndividuos; ++i) {
        m_individuos.push_back(Individuo{nVariables, limites});

        // TODO Evaluar individuos
    }
}

void Enjambre::evaluarFitness()
{
    for (Individuo& ind : m_individuos) {
        double aptitud = m_fitness(ind);

        if (aptitud > ind.aptitudMejorLocal) {
            ind.setMejorLocal(ind.getPosicion());
            ind.aptitudMejorLocal = aptitud;

            if (aptitud > m_mejorAptitud) {
                m_mejorGlobal = ind;
                m_mejorAptitud = aptitud;
            }
        }
    }
}
