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

        Poblacion<8, 1> p{limites,
                          fitness,
                          /*individuos =*/40,
                          /*generaciones =*/500,
                          /*umbral =*/100};

        p.evaluarPoblacion();
        cout << "Función x * sin(sqrt(abs(x)))" << endl
             << "=============================" << endl
             << "Antes de iniciar el entrenamiento" << endl
             << "Mejor individuo: x = " << p.mejorIndividuo().fenotipo().at(0) << endl
             << "Mejor fitness: " << p.mejorFitness() << endl
             << "Fitness promedio: " << p.fitnessPromdedio() << endl;

        {
            const auto& individuos = p.individuos();
            mat puntos(individuos.size(), 2);

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
        string stopChar = "";

        for (generacion = 0; generacion < 500 && !p.termino(); ++generacion) {
            p.evolucionar(1);

            if (stopChar != "s" || p.termino()) {
                const auto& individuos = p.individuos();
                mat puntos(individuos.size(), 2);

                for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                    puntos.at(i, 0) = individuos.at(i).fenotipo().at(0);
                    puntos.at(i, 1) = fitness(individuos.at(i).fenotipo());
                }

                gp << "plot x * sin(sqrt(abs(x))) notitle, "
                   << gp.file1d(puntos) << " notitle with points lw 1.5" << endl;

                if (!p.termino())
                    getline(cin, stopChar);
            }
        }

        cout << "\nLuego de finalizar el entrenamiento" << endl
             << "Mejor individuo: x = " << p.mejorIndividuo().fenotipo().at(0) << endl
             << "Mejor fitness: " << p.mejorFitness() << endl
             << "Fitness promedio: " << p.fitnessPromdedio() << endl
             << "Generaciones evolucionadas: " << generacion << endl;
        getchar();
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

        Poblacion<8, 1> p{limites,
                          fitness,
                          /*individuos =*/40,
                          /*generaciones =*/500,
                          /*umbral =*/100};

        p.evaluarPoblacion();
        cout << "\nFunción -x - 5 * sin(3 * x) - 8 * cos(5 * x)" << endl
             << "=============================" << endl
             << "Antes de iniciar el entrenamiento" << endl
             << "Mejor individuo: x = " << p.mejorIndividuo().fenotipo().at(0) << endl
             << "Mejor fitness: " << p.mejorFitness() << endl
             << "Fitness promedio: " << p.fitnessPromdedio() << endl;

        {
            const auto& individuos = p.individuos();
            mat puntos(individuos.size(), 2);

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
        string stopChar = "";

        for (generacion = 0; generacion < 500 && !p.termino(); ++generacion) {
            p.evolucionar(1);

            if (stopChar != "s" || p.termino()) {
                const auto& individuos = p.individuos();
                mat puntos(individuos.size(), 2);

                for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                    puntos.at(i, 0) = individuos.at(i).fenotipo().at(0);
                    puntos.at(i, 1) = fitness(individuos.at(i).fenotipo());
                }

                gp << "plot -x - 5 * sin(3 * x) - 8 * cos(5 * x) notitle, "
                   << gp.file1d(puntos) << " notitle with points lw 1.5" << endl;

                if (!p.termino())
                    getline(cin, stopChar);
            }
        }

        cout << "\nLuego de finalizar el entrenamiento" << endl
             << "Mejor individuo: x = " << p.mejorIndividuo().fenotipo().at(0) << endl
             << "Mejor fitness: " << p.mejorFitness() << endl
             << "Fitness promedio: " << p.fitnessPromdedio() << endl
             << "Generaciones evolucionadas: " << generacion << endl;
        getchar();
    }

    // -------------------------------
    // Inciso c
    // -------------------------------
    {
        const array<double, 4> limites = {{-100, 100, -100, 100}};

        auto fitness = [](array<double, 2> fenotipo) -> double {
            double x = fenotipo.at(0);
            double y = fenotipo.at(1);

            double parte1 = pow(x * x + y * y, 0.25);
            double parte2 = pow(sin(50 * pow(x * x + y * y, 0.1)), 2) + 1;

            return -parte1 * parte2;
        };

        Poblacion<14, 2> p{limites,
                           fitness,
                           /*individuos =*/300,
                           /*generaciones =*/1000,
                           /*umbral =*/200};

        p.evaluarPoblacion();
        cout << "\n-(pow(x * x + y * y, 0.25) * (pow(sin(50 * pow(x * x + y * y, 0.1)), 2)) + 1)" << endl
             << "===============================================================================" << endl
             << "Antes de iniciar el entrenamiento" << endl
             << "Mejor individuo: "
             << "x = " << p.mejorIndividuo().fenotipo().at(0)
             << " y = " << p.mejorIndividuo().fenotipo().at(1) << endl
             << "Mejor fitness: " << p.mejorFitness() << endl
             << "Fitness promedio: " << p.fitnessPromdedio() << endl;

        {
            const auto& individuos = p.individuos();
            mat puntos(individuos.size(), 3);

            for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                puntos.at(i, 0) = individuos.at(i).fenotipo().at(0);
                puntos.at(i, 1) = individuos.at(i).fenotipo().at(1);
                puntos.at(i, 2) = fitness(individuos.at(i).fenotipo());
            }

            gp << "set xlabel 'x' font ',10'" << endl
               << "set ylabel 'y' font ',10'" << endl
               << "set xrange [-100:100]" << endl
               << "set yrange [-100:100]" << endl
               << "set grid lw 2" << endl
               << "set samples 60" << endl
               << "set isosamples 60" << endl
               << "set pm3d depthorder hidden3d" << endl
               << "splot (-((x * x + y * y)**(0.25) * (sin(50 * (x * x + y * y)**(0.1))**2 + 1))) palette notitle, "
               << gp.file1d(puntos) << " notitle with points lw 1.5" << endl;

            getchar();
        }

        int generacion;
        string stopChar = "";

        for (generacion = 0; generacion < 500 && !p.termino(); ++generacion) {
            p.evolucionar(1);

            if (stopChar != "s" || p.termino()) {
                const auto& individuos = p.individuos();
                mat puntos(individuos.size(), 3);

                for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                    puntos.at(i, 0) = individuos.at(i).fenotipo().at(0);
                    puntos.at(i, 1) = individuos.at(i).fenotipo().at(1);
                    puntos.at(i, 2) = fitness(individuos.at(i).fenotipo());
                }

                gp << "splot (-((x * x + y * y)**(0.25) * (sin(50 * (x * x + y * y)**(0.1))**2 + 1))) palette notitle, "
                   << gp.file1d(puntos) << " notitle with points lw 1.5" << endl;

                if (!p.termino())
                    getline(cin, stopChar);
            }
        }

        cout << "\nLuego de finalizar el entrenamiento" << endl
             << "Mejor individuo: "
             << "x = " << p.mejorIndividuo().fenotipo().at(0)
             << " y = " << p.mejorIndividuo().fenotipo().at(1) << endl
             << "Mejor fitness: " << p.mejorFitness() << endl
             << "Fitness promedio: " << p.fitnessPromdedio() << endl
             << "Cantidad de generaciones evolucionadas: " << generacion << endl;
        getchar();
    }

    return 0;
}
