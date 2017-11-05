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
          const array<double, nVariables * 2>& limites>
struct Individuo {
    // Individuo inicializado al azar
    Individuo()
        : genotipo{unsigned(randi(1, distr_param(0.0, pow(2, nBits * nVariables) - 1)).at(0))}
    {
    }

    Individuo(bitset<nBits * nVariables> t_genotipo)
        : genotipo{t_genotipo} {};

    array<double, nVariables> fenotipo() const
    {
        return decodificar<nBits, nVariables, limites>(genotipo);
    }

    void mutar()
    {
        genotipo.flip(randi(1, distr_param(0.0, nBits - 1)).at(0));
    }

    bitset<nBits * nVariables> genotipo;
};

// ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧
// Cuidado al seguir leyendo debajo!
// ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧ ⛧

template <unsigned int nBits,
          unsigned int nVariables,
          const array<double, nVariables * 2>& limites,
          std::function<double(array<double, nVariables>)>& fitness>
class Poblacion {
    using I = Individuo<nBits, nVariables, limites>;

public:
    Poblacion(int nIndividuos,
              int nGeneraciones,
              int umbral);

    bool evaluarPoblacion();
    int evolucionar();
    vector<I> seleccionarPadres();
    vector<I> hacerCruzas(const vector<I>& padres, int nHijos);

    const vector<I>& individuos() const { return m_individuos; };
    double mejorFitness() const { return m_mejorAptitud; };
    double fitnessPromdedio() const;

    static pair<I, I> cruzar(const I& padre1, const I& padre2, int puntoCruza);

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
    for (const I& ind : m_individuos) {
        double aptitud = fitness(ind.fenotipo());

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
int Poblacion<nBits,
              nVariables,
              limites,
              fitness>::
    evolucionar()
{
    evaluarPoblacion();

    int generacion;
    for (generacion = 0; generacion < m_nGeneraciones; ++generacion) {
        vector<I> nuevaGeneracion;
        // 1 - Rescatar el mejor individuo y meterlo en la siguiente generacion
        nuevaGeneracion.push_back(m_mejorIndividuo);

        // 2 - Seleccionar los padres
        vector<I> padres = seleccionarPadres();
        copy(padres.begin(), padres.end(), back_inserter(nuevaGeneracion));

        // 3 - Hacer cruzas
        vector<I> hijos = hacerCruzas(padres, m_nIndividuos * 0.7);

        // 4 - Hacer mutaciones
        for (I& h : hijos)
            if (randu(1).at(0, 0) <= 0.01)
                h.mutar();

        copy(hijos.begin(), hijos.end(), back_inserter(nuevaGeneracion));
        m_individuos = nuevaGeneracion;

        // 5 - Evaluar la poblacion y ver si se dio el criterio de parada
        if (!evaluarPoblacion())
            break;
    }

    return generacion;
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
    const int nPadres = 0.3 * m_nIndividuos - 1;
    const int k = 3;

    for (int i = 0; i < nPadres; ++i) {
        const uvec indices = shuffle(linspace<uvec>(0, m_nIndividuos - 1, m_nIndividuos));

        vector<I> candidatos;
        vec aptitudes(k);
        for (int j = 0; j < k; ++j) {
            candidatos.push_back(m_individuos.at(indices(j)));
            aptitudes(j) = fitness(candidatos.at(j).fenotipo());
        }

        I mejor = m_individuos.at(indices(aptitudes.index_max()));
        result.push_back(mejor);
    }

    return result;
}

template <unsigned int nBits,
          unsigned int nVariables,
          const array<double, nVariables * 2>& limites,
          std::function<double(array<double, nVariables>)>& fitness>
auto Poblacion<nBits,
               nVariables,
               limites,
               fitness>::
    cruzar(const I& padre1, const I& padre2, int puntoCruza)
        -> pair<I, I>
{
    I hijo1, hijo2;

    for (int i = 0; i < int(padre1.genotipo.size()); ++i) {
        // Nota: El operador de acceso ([]) a un bitset los accede desde
        // el bit menos significativo al más significativo.

        if (i < puntoCruza) { // Parte izquierda del cromosoma
            hijo1.genotipo[i] = padre1.genotipo[i];
            hijo2.genotipo[i] = padre2.genotipo[i];
        }
        else { // Parte derecha del cromosoma
            hijo1.genotipo[i] = padre2.genotipo[i];
            hijo2.genotipo[i] = padre1.genotipo[i];
        }
    }

    return {hijo1, hijo2};
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
    vector<I> hijos;

    for (int i = 0; i < nHijos; i += 2) {
        if (padresAux.empty()) {
            padresAux = padres;
            random_shuffle(padresAux.begin(), padresAux.end());
        }

        I padre1 = padresAux.back();
        padresAux.pop_back();
        I padre2 = padresAux.back();
        padresAux.pop_back();

        int puntoCruza = randi(1, distr_param(1, nBits - 1))(0);

        I hijo1, hijo2;

        if (randu(1).at(0, 0) <= 0.9) // 90% de probabilidad de cruza
            tie(hijo1, hijo2) = cruzar(padre1, padre2, puntoCruza);
        else {
            hijo1 = padre1;
            hijo2 = padre2;
        }

        hijos.push_back(hijo1);
        hijos.push_back(hijo2);
    }

    return hijos;
}
template <unsigned int nBits,
          unsigned int nVariables,
          const array<double, nVariables * 2>& limites,
          std::function<double(array<double, nVariables>)>& fitness>
double Poblacion<nBits,
                 nVariables,
                 limites,
                 fitness>::
    fitnessPromdedio() const
{
    double suma = 0;

    for (const I& ind : m_individuos) {
        double aptitud = fitness(ind.fenotipo());
        suma += aptitud;
    }

    return suma / m_individuos.size();
}
