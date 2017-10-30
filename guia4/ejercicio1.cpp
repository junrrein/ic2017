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
        m_cromosoma = bitset<16>{unsigned(randi(1, distr_param(0.0, pow(2, m_cromosoma.size()) - 1))(0))};
    }

    array<double, 1> fenotipo() const override
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

    Poblacion<IndividuoF1, 16, 1> p{[](IndividuoF1) { return 0; },
                                    100};

    for (const auto& ind : p.individuos())
        cout << ind.fenotipo().at(0) << endl;

    return 0;
}
