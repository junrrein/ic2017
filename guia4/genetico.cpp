#include <armadillo>
#include <bitset>
#include <functional>

using namespace arma;
using namespace std;

template <unsigned int nBits>
bitset<nBits> funcionCodificacion(double fenotipo,
                                  pair<double, double> limites)
{
    double rango = limites.second - limites.first;
    double factorConversion = (pow(2, nBits) - 1) / rango;

    unsigned int convertido = round((fenotipo - limites.first) * factorConversion);

    return bitset<nBits>{convertido};
}

template <unsigned int nBits, unsigned int nVariables>
bitset<nBits * nVariables> funcionCodificacion(array<double, nVariables> fenotipos,
                                               array<pair<double, double>, nVariables> limites)
{
    ostringstream ost;
    for (unsigned int i = 0; i < fenotipos.size(); ++i) {
        ost << funcionCodificacion<nBits>(fenotipos.at(i), limites.at(i));
    }

    bitset<nBits * limites.size()> result;
    istringstream ist{ost.str()};
    ist >> result;

    return result;
}

template <unsigned int nBits>
double funcionDecodificacion(bitset<nBits> genotipo,
                             pair<double, double> limites)
{
    double rango = limites.second - limites.first;
    double factorConversion = rango / (pow(2, nBits) - 1);

    double convertido = genotipo.to_ulong() * factorConversion + limites.first;

    return convertido;
}

template <unsigned int nBits, unsigned int nVariables>
array<double, nVariables> funcionDecodificacion(bitset<nBits * nVariables> genotipo,
                                                array<pair<double, double>, nVariables> limites)
{
    istringstream ist{genotipo.to_string()};
    array<double, nVariables> result;

    for (unsigned int i = 0; i < result.size(); ++i) {
        bitset<nBits> trozoGenotipo;
        ist >> trozoGenotipo;
        result.at(i) = funcionDecodificacion<nBits>(trozoGenotipo, limites.at(i));
    }

    return result;
}

// Genotipo con un único cromosoma
template <unsigned int nBits, unsigned int nVariables>
class Individuo {
public:
    virtual array<double, nVariables> fenotipo() = 0;
    bitset<nBits * nVariables>& genotipo() { return m_cromosoma; };

protected:
    bitset<nBits * nVariables> m_cromosoma;
};

template <typename I,
          unsigned int nBits,
          unsigned int nVariables>
class Poblacion {
    typedef typename std::enable_if<std::is_base_of<Individuo<nBits, nVariables>, I>::value>::type check;

public:
    Poblacion(std::function<double(I)> funcionFitness,
              int nIndividuos);

private:
    std::function<double(I)> m_funcionFitness;
    const int m_nIndividuos;
    vector<I> m_individuos;
};

template <typename I,
          unsigned int nBits,
          unsigned int nVariables>
Poblacion<I, nBits, nVariables>::Poblacion(std::function<double(I)> funcionFitness,
                                           int nIndividuos)
    : m_funcionFitness{funcionFitness}
    , m_nIndividuos{nIndividuos}
{
    for (int i = 0; i < nIndividuos; ++i)
        m_individuos.push_back(I{});
}
