#include "genetico.cpp"
#include "enjambre.cpp"

int main()
{
    // Algoritmo genético
    {
        const array<double, 4> limites = {{-100, 100, -100, 100}};

        auto fitness = [](array<double, 2> fenotipo) -> double {
            double x = fenotipo.at(0);
            double y = fenotipo.at(1);

            double parte1 = pow(x * x + y * y, 0.25);
            double parte2 = pow(sin(50 * pow(x * x + y * y, 0.1)), 2) + 1;

            return -parte1 * parte2;
        };

        vec mejoresSoluciones(20);

        wall_clock reloj;
        reloj.tic();

        for (int i = 0; i < 20; ++i) {
            genetico::Poblacion<12, 2> p{limites,
                                         fitness,
                                         /*individuos =*/150,
                                         /*generaciones =*/500,
                                         /*umbral =*/100};

            int generacion;
            for (generacion = 0; generacion < 500 && !p.termino(); ++generacion)
                p.evolucionar(1);

            mejoresSoluciones.at(i) = p.mejorFitness();
        }

        double tiempoGenetico;
        tiempoGenetico = reloj.toc();

        cout << "Tiempo para algoritmo genético (20 corridas): "
             << tiempoGenetico << " segundos" << endl
             << "Fitness promedio de la solución encontrada: "
             << mean(mejoresSoluciones) << endl;
    }

    // Algoritmo enjambre
    {
        const vector<pair<double, double>> limites = {{-100, 100}, {-100, 100}};

        auto fitness = [](enjambre::Individuo ind) -> double {
            double x = ind.getPosicion().at(0);
            double y = ind.getPosicion().at(1);

            double parte1 = pow(x * x + y * y, 0.25);
            double parte2 = pow(sin(50 * pow(x * x + y * y, 0.1)), 2) + 1;

            return -parte1 * parte2;
        };

        vec mejoresSoluciones(20);

        wall_clock reloj;
        reloj.tic();

        for (int i = 0; i < 20; ++i) {
            enjambre::Enjambre e{fitness,
                                 limites,
                                 /*c1 =*/0.5,
                                 /*c2 =*/0.1,
                                 /*individuos =*/10,
                                 /*umbral =*/50};

            int generacion;
            for (generacion = 0; generacion < 500 && !e.termino(); ++generacion)
                e.epoca();

            mejoresSoluciones.at(i) = e.mejorGlobal().aptitudMejorLocal;
        }

        double tiempoEnjambre;
        tiempoEnjambre = reloj.toc();

        cout << "\nTiempo para algoritmo de enjambre (20 corridas): "
             << tiempoEnjambre << " segundos" << endl
             << "Fitness promedio de la solución encontrada: "
             << mean(mejoresSoluciones) << endl;
    }

    return 0;
}
