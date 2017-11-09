#include "enjambre.cpp"
#include <gnuplot-iostream.h>

using namespace enjambre;

int main()
{
    arma_rng::set_seed_random();
    Gnuplot gp;

    // -------------------------------
    // Inciso a
    // -------------------------------
    {
        const vector<pair<double, double>> limites = {{-512, 512}};

        auto fitness = [](const Individuo& ind) -> double {
            double x = ind.getPosicion().at(0);

            return x * sin(sqrt(abs(x)));
        };

        Enjambre e{fitness,
                   limites,
                   /*c1 =*/0.5,
                   /*c2 =*/0.1,
                   /*individuos =*/10,
                   /*umbral =*/50};

        e.evaluarPoblacion();
        cout << "Función x * sin(sqrt(abs(x)))" << endl
             << "=============================" << endl
             << "Antes de iniciar el entrenamiento" << endl
             << "Mejor individuo: x = " << e.mejorGlobal().getPosicion().at(0) << endl
             << "Mejor fitness: " << e.mejorGlobal().aptitudMejorLocal << endl
             << "Fitness promedio: " << e.fitnessPromedio() << endl;

        {
            const auto& individuos = e.individuos();
            mat puntos(individuos.size(), 2);

            for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                puntos.at(i, 0) = individuos.at(i).getPosicion().at(0);
                puntos.at(i, 1) = fitness(individuos.at(i));
            }

            gp << "set title 'Bandada de pájaros - inciso (a)' font ',11'" << endl
               << "set xlabel 'x' font ',10'" << endl
               << "set ylabel 'aptitud' font ',10'" << endl
               << "set xrange [-512:512]" << endl
               << "set grid" << endl
               << "set key box opaque center top" << endl
               << "set samples 500" << endl
               << "plot x * sin(sqrt(abs(x))) title 'Función de aptitud', "
               << gp.file1d(puntos) << " title 'Individuos' with points lw 1.5" << endl;

            getchar();
        }

        int generacion;
        string stopChar = "";

        for (generacion = 0; generacion < 500 && !e.termino(); ++generacion) {
            e.epoca();

            if (stopChar != "s" || e.termino()) {
                const auto& individuos = e.individuos();
                mat puntos(individuos.size(), 2);

                for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                    puntos.at(i, 0) = individuos.at(i).getPosicion().at(0);
                    puntos.at(i, 1) = fitness(individuos.at(i));
                }

                gp << "plot x * sin(sqrt(abs(x))) title 'Función de aptitud', "
                   << gp.file1d(puntos) << " title 'Individuos' with points lw 1.5" << endl;

                if (!e.termino())
                    getline(cin, stopChar);
            }
        }

        cout << "\nLuego de finalizar el entrenamiento" << endl
             << "Mejor individuo: x = " << e.mejorGlobal().getPosicion().at(0) << endl
             << "Mejor fitness: " << e.mejorGlobal().aptitudMejorLocal << endl
             << "Fitness promedio: " << e.fitnessPromedio() << endl
             << "Generaciones evolucionadas: " << generacion << endl;
        getchar();
    }

    // -------------------------------
    // Inciso b
    // -------------------------------
    {
        const vector<pair<double, double>> limites = {{0, 20}};

        auto fitness = [](const Individuo& ind) -> double {
            double x = ind.getPosicion().at(0);

            return -x - 5 * sin(3 * x) - 8 * cos(5 * x);
        };

        Enjambre e{fitness,
                   limites,
                   /*c1 =*/0.5,
                   /*c2 =*/0.1,
                   /*individuos =*/10,
                   /*umbral =*/50};

        e.evaluarPoblacion();
        cout << "\nFunción -x - 5 * sin(3 * x) - 8 * cos(5 * x)" << endl
             << "=============================" << endl
             << "Antes de iniciar el entrenamiento" << endl
             << "Mejor individuo: x = " << e.mejorGlobal().getPosicion().at(0) << endl
             << "Mejor fitness: " << e.mejorGlobal().aptitudMejorLocal << endl
             << "Fitness promedio: " << e.fitnessPromedio() << endl;

        {
            const auto& individuos = e.individuos();
            mat puntos(individuos.size(), 2);

            for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                puntos.at(i, 0) = individuos.at(i).getPosicion().at(0);
                puntos.at(i, 1) = fitness(individuos.at(i));
            }

            gp << "set title 'Bandada de pájaros - inciso (b)' font ',11'" << endl
               << "set xlabel 'x' font ',10'" << endl
               << "set ylabel 'aptitud' font ',10'" << endl
               << "set xrange [0:20]" << endl
               << "set grid" << endl
               << "set key box opaque center top" << endl
               << "plot -x - 5 * sin(3 * x) - 8 * cos(5 * x) title 'Función de aptitud', "
               << gp.file1d(puntos) << " title 'Individuos' with points lw 1.5" << endl;

            getchar();
        }

        int generacion;
        string stopChar = "";

        for (generacion = 0; generacion < 500 && !e.termino(); ++generacion) {
            e.epoca();

            if (stopChar != "s" || e.termino()) {
                const auto& individuos = e.individuos();
                mat puntos(individuos.size(), 2);

                for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                    puntos.at(i, 0) = individuos.at(i).getPosicion().at(0);
                    puntos.at(i, 1) = fitness(individuos.at(i));
                }

                gp << "plot -x - 5 * sin(3 * x) - 8 * cos(5 * x) title 'Función de aptitud', "
                   << gp.file1d(puntos) << " title 'Individuos' with points lw 1.5" << endl;

                if (!e.termino())
                    getline(cin, stopChar);
            }
        }

        cout << "\nLuego de finalizar el entrenamiento" << endl
             << "Mejor individuo: x = " << e.mejorGlobal().getPosicion().at(0) << endl
             << "Mejor fitness: " << e.mejorGlobal().aptitudMejorLocal << endl
             << "Fitness promedio: " << e.fitnessPromedio() << endl
             << "Generaciones evolucionadas: " << generacion << endl;
        getchar();
    }

    // -------------------------------
    // Inciso c
    // -------------------------------
    {
        const vector<pair<double, double>> limites = {{-100, 100}, {-100, 100}};

        auto fitness = [](const Individuo& ind) -> double {
            double x = ind.getPosicion().at(0);
            double y = ind.getPosicion().at(1);

            double parte1 = pow(x * x + y * y, 0.25);
            double parte2 = pow(sin(50 * pow(x * x + y * y, 0.1)), 2) + 1;

            return -parte1 * parte2;
        };

        Enjambre e{fitness,
                   limites,
                   /*c1 =*/0.5,
                   /*c2 =*/0.1,
                   /*individuos =*/10,
                   /*umbral =*/50};

        e.evaluarPoblacion();
        cout << "\n-(pow(x * x + y * y, 0.25) * (pow(sin(50 * pow(x * x + y * y, 0.1)), 2)) + 1)" << endl
             << "=============================" << endl
             << "Antes de iniciar el entrenamiento" << endl
             << "Mejor individuo: "
             << "x = " << e.mejorGlobal().getPosicion().at(0)
             << " y = " << e.mejorGlobal().getPosicion().at(1) << endl
             << "Mejor fitness: " << e.mejorGlobal().aptitudMejorLocal << endl
             << "Fitness promedio: " << e.fitnessPromedio() << endl;

        {
            const auto& individuos = e.individuos();
            mat puntos(individuos.size(), 3);

            for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                puntos.at(i, 0) = individuos.at(i).getPosicion().at(0);
                puntos.at(i, 1) = individuos.at(i).getPosicion().at(1);
                puntos.at(i, 2) = fitness(individuos.at(i));
            }

            gp << "set title 'Algoritmo genético - inciso (c)' font ',11'" << endl
               << "set xlabel 'x' font ',10'" << endl
               << "set ylabel 'y' font ',10'" << endl
               << "set zlabel 'aptitud' font ',10'" << endl
               << "set xrange [-100:100]" << endl
               << "set yrange [-100:100]" << endl
               << "set grid lw 2" << endl
               << "set key box opaque center top" << endl
               << "set samples 60" << endl
               << "set isosamples 60" << endl
               << "set pm3d depthorder hidden3d" << endl
               << "splot (-((x * x + y * y)**(0.25) * (sin(50 * (x * x + y * y)**(0.1))**2 + 1))) palette title 'Función de aptitud', "
               << gp.file1d(puntos) << " title 'Individuos' with points lw 1.5" << endl;
            getchar();
        }

        int generacion;
        string stopChar = "";

        for (generacion = 0; generacion < 500 && !e.termino(); ++generacion) {
            e.epoca();

            if (stopChar != "s" || e.termino()) {
                const auto& individuos = e.individuos();
                mat puntos(individuos.size(), 3);

                for (unsigned int i = 0; i < puntos.n_rows; ++i) {
                    puntos.at(i, 0) = individuos.at(i).getPosicion().at(0);
                    puntos.at(i, 1) = individuos.at(i).getPosicion().at(1);
                    puntos.at(i, 2) = fitness(individuos.at(i));
                }

                gp << "splot (-((x * x + y * y)**(0.25) * (sin(50 * (x * x + y * y)**(0.1))**2 + 1))) palette title 'Función de aptitud', "
                   << gp.file1d(puntos) << " title 'Individuos' with points lw 1.5" << endl;

                if (!e.termino())
                    getline(cin, stopChar);
            }
        }

        cout << "\nLuego de finalizar el entrenamiento" << endl
             << "Mejor individuo: "
             << "x = " << e.mejorGlobal().getPosicion().at(0)
             << " y = " << e.mejorGlobal().getPosicion().at(1) << endl
             << "Mejor fitness: " << e.mejorGlobal().aptitudMejorLocal << endl
             << "Fitness promedio: " << e.fitnessPromedio() << endl
             << "Generaciones evolucionadas: " << generacion << endl;
        getchar();
    }

    return 0;
}
