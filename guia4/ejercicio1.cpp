#include "genetico.cpp"

array<double, 2> limitesF1 = {{-512, 512}};
std::function<double(array<double, 1>)> fitness1
    = [](array<double, 1> fenotipo) -> double {
    double x = fenotipo.at(0);

    return x * sin(sqrt(abs(x)));
};

int main()
{
    arma_rng::set_seed_random();

    Poblacion<16, 1, limitesF1, fitness1> p{/*individuos =*/100,
                                            /*generaciones =*/500,
                                            /*umbral =*/50};

    for (const auto& ind : p.individuos())
        cout << ind.fenotipo().at(0) << endl;

    p.evaluarPoblacion();
    cout << "Individuo con mejor fitness: " << p.mejorFitness();

    auto padres = p.seleccionarPadres();

    return 0;
}
