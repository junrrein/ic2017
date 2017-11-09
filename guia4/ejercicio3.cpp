#include "costo_uniforme.cpp"
#include "colonia_hormigas.cpp"
#include "../config.hpp"
#include <atomic>

int main()
{
    arma_rng::set_seed_random();
    wall_clock reloj;

    const string rutaArchivo7 = config::sourceDir + "/guia4/datos/7cities.csv";
    const string rutaArchivo10 = config::sourceDir + "/guia4/datos/10cities.csv";

    ArbolBusqueda a7{rutaArchivo7};
    ArbolBusqueda a10{rutaArchivo10};

    reloj.tic();
    Camino solucion7 = a7.hacerBusqueda();
    double tiempo7arbol = reloj.toc();

    reloj.tic();
    Camino solucion10 = a10.hacerBusqueda();
    double tiempo10arbol = reloj.toc();

    cout << "Usando árbol de búsqueda con costo uniforme" << endl
         << "===========================================" << endl
         << "\nProblema de las 7 ciudades" << endl
         << "Solución encontrada:" << endl
         << solucion7
         << "Tiempo: " << tiempo7arbol << endl
         << "\nProblema de las 10 ciudades" << endl
         << "Solución encontrada:" << endl
         << solucion10
         << "Tiempo: " << tiempo10arbol << endl;

    // Bloque hormigas 7 ciudades
    {
        vec distancias(100);
        atomic<int> exitos{0};
        atomic<int> optimos{0};

        double tiempo7colonia = 0;

        //#pragma omp parallel for
        for (int i = 0; i < 1; ++i) {
            ColoniaHormigas c7(rutaArchivo7,
                               /*nHormigas =*/40,
                               /*nEpocas =*/500,
                               /*sigma_cero =*/0.5,
                               /*alpha =*/1.5,
                               /*beta =*/1,
                               /*tasaEvaporacion =*/0.2,
                               /*Q =*/4);

            try {
                reloj.tic();
                Hormiga solucion = c7.encontrarSolucion();
                tiempo7colonia = reloj.toc();

                const double distancia = solucion.costoCamino;
                distancias(i) = distancia;
                ++exitos;

                if (abs(distancia - 30.7188) < 1e-3)
                    ++optimos;
            }
            catch (...) {
            }
        }

        cout << "\nUsando colonia de hormigas (100 colonias x 100 hormigas)" << endl
             << "==========================================================" << endl
             << "\nProblema de las 7 ciudades" << endl
             << "Resultados obtenidos:" << endl
             << "Distancia promedio de solución: " << distancias(0) << endl
             << "Converge " << exitos << " veces" << endl
             << "Converge a la mejor solución " << optimos << " veces" << endl
             << "Tiempo: " << tiempo7colonia << endl;
    }

    // Bloque hormigas 10 ciudades
    {
        vec distancias(100);
        atomic<int> exitos{0};
        atomic<int> optimos{0};

        double tiempo10colonia = 0;

        //#pragma omp parallel for
        for (int i = 0; i < 1; ++i) {
            ColoniaHormigas c10(rutaArchivo10,
                                /*nHormigas =*/40,
                                /*nEpocas =*/500,
                                /*sigma_cero =*/0.5,
                                /*alpha =*/1.5,
                                /*beta =*/2,
                                /*tasaEvaporacion =*/0.2,
                                /*Q =*/2);

            try {
                reloj.tic();
                Hormiga solucion = c10.encontrarSolucion();
                tiempo10colonia = reloj.toc();

                const double distancia = solucion.costoCamino;
                distancias(i) = distancia;
                ++exitos;

                if (abs(distancia - 298.0) < 1.0)
                    ++optimos;
            }
            catch (...) {
            }
        }

        cout << "\nProblema de las 10 ciudades" << endl
             << "Resultados obtenidos:" << endl
             << "Distancia promedio de solución: " << distancias(0) << endl
             << "Converge " << exitos << " veces" << endl
             << "Converge a la mejor solución " << optimos << " veces" << endl
             << "Tiempo: " << tiempo10colonia << endl;
    }

    return 0;
}
