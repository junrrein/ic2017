#include "genetico.cpp"

class IndividuoF1 : public Individuo<16, 1> {
public:
    IndividuoF1(double fenotipo)
    {
        m_cromosoma = funcionCodificacion<16>(fenotipo,
                                              m_limites);
    }

    IndividuoF1()
    {
        m_cromosoma = bitset<16>{unsigned(randi(1, distr_param(m_limites.first, m_limites.second))(0))};
    }

    array<double, 1> fenotipo() override
    {
        return {{funcionDecodificacion<16>(m_cromosoma, m_limites)}};
    }

    static pair<double, double> m_limites;

private:
};

pair<double, double> IndividuoF1::m_limites = {-512, 512};

int main()
{
    arma_rng::set_seed_random();

    IndividuoF1 i1{-512};
    IndividuoF1 i2{512};

    cout << i1.genotipo() << endl
         << i2.genotipo() << endl;

    cout << i1.fenotipo().at(0) << endl
         << i2.fenotipo().at(0) << endl;

    Poblacion<IndividuoF1, 16, 1> p{[](IndividuoF1) { return 0; },
                                    100};

    return 0;
}
