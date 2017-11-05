#include "genetico.cpp"
#include <gnuplot-iostream.h>

int main()
{
    arma_rng::set_seed_random();
    Gnuplot gp;

    // -------------------------------
    // Inciso a
    // -------------------------------
    {
        const array<double, 2> limites = {{-512, 512}};

        auto fitness = [](array<double, 1> fenotipo) -> double {
            double x = fenotipo.at(0);

            return x * sin(sqrt(abs(x)));
        };

        Poblacion<16, 1> p{limites,
                           fitness,
                           /*individuos =*/40,
                           /*generaciones =*/500,
                           /*umbral =*/50};

        p.evaluarPoblacion();
        cout << "Función x * sin(sqrt(abs(x)))" << endl
             << "=============================" << endl
             << "Antes de iniciar el entrenamiento" << endl
             << "Mejor individuo: x = " << p.mejorIndividuo().fenotipo().at(0) << endl
             << "Mejor fitness: " << p.mejorFitness() << endl
             << "Fitness promedio: " << p.fitnessPromdedio() << endl;

        {
            const vector<Individuo<16, 1>> individuos = p.individuos();
            mat puntos(40, 2);

            for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                puntos.at(i, 0) = individuos.at(i).fenotipo().at(0);
                puntos.at(i, 1) = fitness(individuos.at(i).fenotipo());
            }

            gp << "set xrange [-512:512]" << endl
               << "set grid" << endl
               << "set samples 500" << endl
               << "plot x * sin(sqrt(abs(x))) notitle, "
               << gp.file1d(puntos) << " notitle with points lw 1.5" << endl;

            getchar();
        }

        int generacion;
        char ch = ' ';

        for (generacion = 0; generacion < 500; ++generacion) {
            p.evolucionar(1);

            if (ch != 's') {
                const vector<Individuo<16, 1>> individuos = p.individuos();
                mat puntos(40, 2);

                for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                    puntos.at(i, 0) = individuos.at(i).fenotipo().at(0);
                    puntos.at(i, 1) = fitness(individuos.at(i).fenotipo());
                }

                gp << "set xrange [-512:512]" << endl
                   << "set grid" << endl
                   << "plot x * sin(sqrt(abs(x))) notitle, "
                   << gp.file1d(puntos) << " notitle with points lw 1.5" << endl;

                ch = getchar();
            }
        }

        cout << "\nLuego de finalizar el entrenamiento" << endl
             << "Mejor individuo: x = " << p.mejorIndividuo().fenotipo().at(0) << endl
             << "Mejor fitness: " << p.mejorFitness() << endl
             << "Fitness promedio: " << p.fitnessPromdedio() << endl
             << "Generaciones evolucionadas: " << generacion << endl;
    }

    // -------------------------------
    // Inciso b
    // -------------------------------
    {
        const array<double, 2> limites = {{0, 20}};

        auto fitness = [](array<double, 1> fenotipo) -> double {
            double x = fenotipo.at(0);

            return -x - 5 * sin(3 * x) - 8 * cos(5 * x);
        };

        Poblacion<16, 1> p{limites,
                           fitness,
                           /*individuos =*/40,
                           /*generaciones =*/500,
                           /*umbral =*/50};

        p.evaluarPoblacion();
        cout << "\nFunción -x - 5 * sin(3 * x) - 8 * cos(5 * x)" << endl
             << "=============================" << endl
             << "Antes de iniciar el entrenamiento" << endl
             << "Mejor individuo: x = " << p.mejorIndividuo().fenotipo().at(0) << endl
             << "Mejor fitness: " << p.mejorFitness() << endl
             << "Fitness promedio: " << p.fitnessPromdedio() << endl;

        {
            const vector<Individuo<16, 1>> individuos = p.individuos();
            mat puntos(40, 2);

            for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                puntos.at(i, 0) = individuos.at(i).fenotipo().at(0);
                puntos.at(i, 1) = fitness(individuos.at(i).fenotipo());
            }

            gp << "set xrange [0:20]" << endl
               << "set grid" << endl
               << "plot -x - 5 * sin(3 * x) - 8 * cos(5 * x) notitle, "
               << gp.file1d(puntos) << " notitle with points lw 1.5" << endl;

            getchar();
        }

        int generacion;
        char ch = ' ';

        for (generacion = 0; generacion < 500; ++generacion) {
            p.evolucionar(1);

            if (ch != 's') {
                const vector<Individuo<16, 1>> individuos = p.individuos();
                mat puntos(40, 2);

                for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                    puntos.at(i, 0) = individuos.at(i).fenotipo().at(0);
                    puntos.at(i, 1) = fitness(individuos.at(i).fenotipo());
                }

                gp << "set xrange [0:20]" << endl
                   << "set grid" << endl
                   << "plot -x - 5 * sin(3 * x) - 8 * cos(5 * x) notitle, "
                   << gp.file1d(puntos) << " notitle with points lw 1.5" << endl;

                ch = getchar();
            }
        }

        cout << "\nLuego de finalizar el entrenamiento" << endl
             << "Mejor individuo: x = " << p.mejorIndividuo().fenotipo().at(0) << endl
             << "Mejor fitness: " << p.mejorFitness() << endl
             << "Fitness promedio: " << p.fitnessPromdedio() << endl
             << "Generaciones evolucionadas: " << generacion << endl;
    }

    return 0;
}
