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

    Poblacion<16, 1, fitness1> p1{limitesF1,
                                  /*individuos =*/40,
                                  /*generaciones =*/500,
                                  /*umbral =*/50};

    p1.evaluarPoblacion();
    cout << "Función x * sin(sqrt(abs(x)))" << endl
         << "Antes de iniciar el entrenamiento" << endl
         << "Mejor individuo: x = " << p1.mejorIndividuo().fenotipo().at(0) << endl
         << "Mejor fitness: " << p1.mejorFitness() << endl
         << "Fitness promedio: " << p1.fitnessPromdedio() << endl;

    int generacion = p1.evolucionar();
    cout << "Luego de finalizar el entrenamiento" << endl
         << "Mejor individuo: x = " << p1.mejorIndividuo().fenotipo().at(0) << endl
         << "Mejor fitness: " << p1.mejorFitness() << endl
         << "Fitness promedio: " << p1.fitnessPromdedio() << endl
         << "Generaciones evolucionadas: " << generacion;

    return 0;
}
