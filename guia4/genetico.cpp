#include <armadillo>
#include <bitset>
#include <functional>

using namespace arma;
using namespace std;

//template <unsigned int nBits>
//bitset<nBits> codificar(double fenotipo,
//                        pair<double, double> limites)
//{
//    double rango = limites.second - limites.first;
//    double factorConversion = (pow(2, nBits) - 1) / rango;

//    unsigned int convertido = round((fenotipo - limites.first) * factorConversion);

//    return bitset<nBits>{convertido};
//}

//template <unsigned int nBits, unsigned int nVariables>
//bitset<nBits * nVariables> codificar(array<double, nVariables> fenotipos,
//                                     array<pair<double, double>, nVariables> limites)
//{
//    ostringstream ost;
//    for (unsigned int i = 0; i < fenotipos.size(); ++i) {
//        ost << codificar<nBits>(fenotipos.at(i), limites.at(i));
//    }

//    bitset<nBits * limites.size()> result;
//    istringstream ist{ost.str()};
//    ist >> result;

//    return result;
//}

template <unsigned int nBits>
double decodificar(bitset<nBits> genotipo,
                   double minimoFenotipo,
                   double maximoFenotipo)
{
    double rango = maximoFenotipo - minimoFenotipo;
    double factorConversion = rango / (pow(2, nBits) - 1);

    double convertido = genotipo.to_ulong() * factorConversion + minimoFenotipo;

    return convertido;
}

template <unsigned int nBits,
          unsigned int nVariables,
          const array<double, nVariables * 2>& limites>
array<double, nVariables> decodificar(bitset<nBits * nVariables> genotipo)
{
    istringstream ist{genotipo.to_string()};
    array<double, nVariables> result;

    for (unsigned int i = 0; i < result.size(); ++i) {
        bitset<nBits> trozoGenotipo;
        ist >> trozoGenotipo;
        result.at(i) = decodificar<nBits>(trozoGenotipo,
                                          limites.at(2 * i),
                                          limites.at(2 * i + 1));
    }

    return result;
}

// Genotipo con un único cromosoma
template <unsigned int nBits,
          unsigned int nVariables,
          const array<double, nVariables * 2>& limites,
          std::function<double(array<double, nVariables>)>& fitness>
class Individuo {
public:
    // Individuo inicializado al azar
    Individuo()
        : m_genotipo{unsigned(randi(1, distr_param(0.0, pow(2, nBits * nVariables) - 1))(0))}
    {
    }

    array<double, nVariables> fenotipo() const
    {
        return decodificar<nBits, nVariables, limites>(m_genotipo);
    }

    double aptitud() const
    {
        return fitness(decodificar<nBits, nVariables, limites>(m_genotipo));
    }

    bitset<nBits * nVariables> m_genotipo;
};

// ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧
// Cuidado al seguir leyendo debajo!
// ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧

template <unsigned int nBits,
          unsigned int nVariables,
          const array<double, nVariables * 2>& limites,
          std::function<double(array<double, nVariables>)>& fitness>
class Poblacion {
    using I = Individuo<nBits, nVariables, limites, fitness>;

public:
    Poblacion(int nIndividuos,
              int nGeneraciones,
              int umbral);

    bool evaluarPoblacion();
    void evolucionar();
    vector<I> seleccionarPadres();
    vector<I> hacerCruzas(const vector<I>& padres, int nHijos);

    const vector<I>& individuos() const { return m_individuos; };
    double mejorFitness() const { return m_mejorAptitud; };

private:
    const int m_nIndividuos;
    vector<I> m_individuos;
    const int m_nGeneraciones;
    const int m_umbral;
    int m_generacion;
    double m_mejorAptitud = numeric_limits<double>::min();
    I m_mejorIndividuo;
    int m_generacionesSinMejora;
};

template <unsigned int nBits,
          unsigned int nVariables,
          const array<double, nVariables * 2>& limites,
          std::function<double(array<double, nVariables>)>& fitness>
Poblacion<nBits,
          nVariables,
          limites,
          fitness>::
    Poblacion(int nIndividuos,
              int nGeneraciones,
              int umbral)
    : m_nIndividuos{nIndividuos}
    , m_nGeneraciones{nGeneraciones}
    , m_umbral{umbral}
{
    for (int i = 0; i < nIndividuos; ++i)
        m_individuos.push_back(I{});
}

// Devuelve true si se cumple la condición de parada por no mejorar fitness
template <unsigned int nBits,
          unsigned int nVariables,
          const array<double, nVariables * 2>& limites,
          std::function<double(array<double, nVariables>)>& fitness>
bool Poblacion<nBits,
               nVariables,
               limites,
               fitness>::
    evaluarPoblacion()
{
    for (const auto& ind : m_individuos) {
        double aptitud = ind.aptitud();

        if (aptitud > m_mejorAptitud) {
            m_mejorAptitud = aptitud;
            m_mejorIndividuo = ind;
            m_generacionesSinMejora = 0;
        }
    }

    if (m_generacionesSinMejora == m_umbral)
        return false;
    else {
        ++m_generacionesSinMejora;
        return true;
    }
}

template <unsigned int nBits,
          unsigned int nVariables,
          const array<double, nVariables * 2>& limites,
          std::function<double(array<double, nVariables>)>& fitness>
void Poblacion<nBits,
               nVariables,
               limites,
               fitness>::
    evolucionar()
{
    evaluarPoblacion();

    for (int i = 0; i < m_nGeneraciones; ++i) {

        vector<I> nuevaGeneracion;
        // 1 - Rescatar el mejor individuo y meterlo en la siguiente generacion
        nuevaGeneracion.push_back(m_mejorIndividuo);

        // 2 - Seleccionar los padres
        vector<I> padres = seleccionarPadres();

        // 3 - Hacer cruzas

        // 4 - Hacer mutaciones

        // 5 - Evaluar la poblacion y ver si se dio el criterio de parada
        if (!evaluarPoblacion())
            break;
    }
}

template <unsigned int nBits,
          unsigned int nVariables,
          const array<double, nVariables * 2>& limites,
          std::function<double(array<double, nVariables>)>& fitness>
auto Poblacion<nBits,
               nVariables,
               limites,
               fitness>::
    seleccionarPadres() -> vector<I>
{
    vector<I> result;
    const int nPadres = 0.3 * m_nIndividuos;
    const int k = 3;

    for (int i = 0; i < nPadres; ++i) {
        const uvec indices = shuffle(linspace<uvec>(0, m_nIndividuos - 1, m_nIndividuos));

        vector<I> candidatos;
        vec aptitudes(k);
        for (int j = 0; j < k; ++j) {
            candidatos.push_back(m_individuos.at(indices(j)));
            aptitudes(j) = candidatos.at(j).aptitud();
        }

        I mejor = m_individuos.at(indices(aptitudes.index_max()));
        result.push_back(mejor);
    }

    return result;
}

template <typename I, unsigned int nBits, unsigned int nVariables>
auto cruzar(const I& padre1, const I& padre2, int puntoCruza) -> pair<I, I>
{
}

template <unsigned int nBits,
          unsigned int nVariables,
          const array<double, nVariables * 2>& limites,
          std::function<double(array<double, nVariables>)>& fitness>
auto Poblacion<nBits,
               nVariables,
               limites,
               fitness>::
    hacerCruzas(const vector<I>& padres, int nHijos) -> vector<I>
{
    vector<I> padresAux;

    for (int i = 0; i < nHijos; i += 2) {
        if (padresAux.empty())
            padresAux = random_shuffle(padres.begin(), padres.end());

        I padre1 = padresAux.back();
        padresAux.pop_back();
        I padre2 = padresAux.back();
        padresAux.pop_back();

        int puntoCruza = randi(1, distr_param(1, nBits - 1))(0);
        // Cruzar padres
    }
}
