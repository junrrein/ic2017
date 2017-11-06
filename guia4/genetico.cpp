#include <armadillo>
#include <bitset>
#include <functional>

using namespace arma;
using namespace std;

template <unsigned int nBits>
bitset<nBits> codificar(double fenotipo,
                        double minimoFenotipo,
                        double maximoFenotipo)
{
    double rango = maximoFenotipo - minimoFenotipo;
    double factorConversion = (pow(2, nBits) - 1) / rango;

    unsigned int convertido = round((fenotipo - minimoFenotipo) * factorConversion);

    return bitset<nBits>{convertido};
}

template <unsigned int nBits, unsigned int nVariables>
bitset<nBits * nVariables> codificar(array<double, nVariables> fenotipo,
                                     const array<double, nVariables * 2>& limites)
{
    ostringstream ost;
    for (unsigned int i = 0; i < nVariables; ++i) {
        ost << codificar<nBits>(fenotipo.at(i),
                                limites.at(2 * i),
                                limites.at(2 * i + 1));
    }

    bitset<nBits * nVariables> result;
    istringstream ist{ost.str()};
    ist >> result;

    return result;
}

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

template <unsigned int nBits, unsigned int nVariables>
array<double, nVariables> decodificar(const bitset<nBits * nVariables>& genotipo,
                                      const array<double, nVariables * 2>& limites)
{
    istringstream ist{genotipo.to_string()};
    array<double, nVariables> result;

    for (unsigned int i = 0; i < nVariables; ++i) {
        bitset<nBits> trozoGenotipo;
        ist >> trozoGenotipo;
        result.at(i) = decodificar<nBits>(trozoGenotipo,
                                          limites.at(2 * i),
                                          limites.at(2 * i + 1));
    }

    return result;
}

// Genotipo con un único cromosoma
template <unsigned int nBits, unsigned int nVariables>
struct Individuo {
    // Individuo inicializado al azar
    Individuo(const array<double, nVariables * 2>& t_limites)
        : genotipo{unsigned(randi(1, distr_param(0.0, pow(2, nBits * nVariables) - 1)).at(0))}
        , limites{t_limites}
    {
    }

    Individuo(bitset<nBits * nVariables> t_genotipo,
              const array<double, nVariables * 2>& t_limites)
        : genotipo{t_genotipo}
        , limites{t_limites}
    {
    }

    array<double, nVariables> fenotipo() const
    {
        return decodificar<nBits, nVariables>(genotipo, limites);
    }

    void mutar()
    {
        genotipo.flip(randi(1, distr_param(0.0, nBits * nVariables - 1)).at(0));
    }

    bitset<nBits * nVariables> genotipo;
    array<double, nVariables * 2> limites;
};

template <unsigned int nBits, unsigned int nVariables>
class Poblacion {
    using I = Individuo<nBits, nVariables>;

public:
    Poblacion(const array<double, nVariables * 2>& limites,
              const std::function<double(array<double, nVariables>)>& fitness,
              int nIndividuos,
              int nGeneraciones,
              int umbral);

    bool evaluarPoblacion();
    void evolucionar(int nGeneraciones);
    vector<I> seleccionarPadres();
    vector<I> hacerCruzas(const vector<I>& padres, int nHijos);

    const vector<I>& individuos() const { return m_individuos; };
    double mejorFitness() const { return m_mejorAptitud; };
    const I& mejorIndividuo() const { return m_mejorIndividuo; }
    bool termino() const { return m_termino; }
    double fitnessPromdedio() const;

    static pair<I, I> cruzar(const I& padre1,
                             const I& padre2,
                             int puntoCruza,
                             const array<double, nVariables * 2>& limites);

private:
    const array<double, nVariables * 2> m_limites;
    const std::function<double(array<double, nVariables>)> m_fitness;
    const int m_nIndividuos;
    vector<I> m_individuos;
    const int m_nGeneraciones;
    const int m_umbral;
    int m_generacion;
    double m_mejorAptitud;
    I m_mejorIndividuo;
    int m_generacionesSinMejora;
    bool m_termino;
};

template <unsigned int nBits, unsigned int nVariables>
Poblacion<nBits, nVariables>::
    Poblacion(const array<double, nVariables * 2>& limites,
              const std::function<double(array<double, nVariables>)>& fitness,
              int nIndividuos,
              int nGeneraciones,
              int umbral)
    : m_limites{limites}
    , m_fitness{fitness}
    , m_nIndividuos{nIndividuos}
    , m_nGeneraciones{nGeneraciones}
    , m_umbral{umbral}
    , m_mejorAptitud{-numeric_limits<double>::max()}
    , m_mejorIndividuo{limites}
    , m_generacionesSinMejora{0}
    , m_termino{false}
{
    for (int i = 0; i < nIndividuos; ++i)
        m_individuos.push_back(I{m_limites});
}

// Devuelve true si se cumple la condición de parada por no mejorar fitness
template <unsigned int nBits, unsigned int nVariables>
bool Poblacion<nBits, nVariables>::
    evaluarPoblacion()
{
    bool mejoro = false;

    for (const I& ind : m_individuos) {
        double aptitud = m_fitness(ind.fenotipo());

        if (aptitud > m_mejorAptitud) {
            m_mejorAptitud = aptitud;
            m_mejorIndividuo = ind;
            mejoro = true;
        }
    }

    return mejoro;
}

template <unsigned int nBits, unsigned int nVariables>
void Poblacion<nBits, nVariables>::
    evolucionar(int nGeneraciones)
{
    // Si nunca se evaluó la población, hacerlo ahora
    if (m_mejorAptitud == numeric_limits<double>::min())
        evaluarPoblacion();

    int generacion;
    for (generacion = 0; generacion < nGeneraciones; ++generacion) {
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
            if (randu(1).at(0, 0) <= 0.1)
                h.mutar();

        copy(hijos.begin(), hijos.end(), back_inserter(nuevaGeneracion));
        m_individuos = nuevaGeneracion;

        // 5 - Evaluar la poblacion y ver si se dio el criterio de parada
        if (evaluarPoblacion())
            m_generacionesSinMejora = 0;
        else {
            if (m_generacionesSinMejora == m_umbral) {
                m_termino = true;
                break;
            }

            ++m_generacionesSinMejora;
        }
    }
}

template <unsigned int nBits, unsigned int nVariables>
auto Poblacion<nBits, nVariables>::
    seleccionarPadres() -> vector<I>
{
    vector<I> result;
    const int nPadres = 0.3 * m_nIndividuos - 1;
    const int k = 2;

    for (int i = 0; i < nPadres; ++i) {
        const uvec indices = shuffle(linspace<uvec>(0, m_nIndividuos - 1, m_nIndividuos));

        vector<I> candidatos;
        vec aptitudes(k);
        for (int j = 0; j < k; ++j) {
            candidatos.push_back(m_individuos.at(indices(j)));
            aptitudes(j) = m_fitness(candidatos.at(j).fenotipo());
        }

        I mejor = m_individuos.at(indices(aptitudes.index_max()));
        result.push_back(mejor);
    }

    return result;
}

template <unsigned int nBits, unsigned int nVariables>
auto Poblacion<nBits, nVariables>::
    cruzar(const I& padre1,
           const I& padre2,
           int puntoCruza,
           const array<double, nVariables * 2>& limites)
        -> pair<I, I>
{
    I hijo1{limites}, hijo2{limites};

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

template <unsigned int nBits, unsigned int nVariables>
auto Poblacion<nBits, nVariables>::
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

        if (padresAux.empty()) {
            padresAux = padres;
            random_shuffle(padresAux.begin(), padresAux.end());
        }
        I padre2 = padresAux.back();
        padresAux.pop_back();

        int puntoCruza = randi(1, distr_param(1, nBits - 1)).at(0);

        I hijo1{m_limites}, hijo2{m_limites};

        if (randu(1).at(0, 0) <= 0.9) // 90% de probabilidad de cruza
            tie(hijo1, hijo2) = cruzar(padre1, padre2, puntoCruza, m_limites);
        else {
            hijo1 = padre1;
            hijo2 = padre2;
        }

        hijos.push_back(hijo1);
        hijos.push_back(hijo2);
    }

    return hijos;
}

template <unsigned int nBits, unsigned int nVariables>
double Poblacion<nBits, nVariables>::
    fitnessPromdedio() const
{
    double suma = 0;

    for (const I& ind : m_individuos) {
        double aptitud = m_fitness(ind.fenotipo());
        suma += aptitud;
    }

    return suma / m_individuos.size();
}
