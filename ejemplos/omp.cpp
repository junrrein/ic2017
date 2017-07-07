#include <armadillo>

using namespace arma;

int main()
{
    vec A(100000000);
    wall_clock clock;

    clock.tic();

    for (unsigned int i = 0; i < A.size(); ++i) {
        A(i) = log(i) * log(i);
    }

    double tiempo1 = clock.toc();

    vec B(100000000);

    clock.tic();

#pragma omp parallel for
    for (unsigned int i = 0; i < A.size(); ++i) {
        B(i) = log(i) * log(i);
    }

    double tiempo2 = clock.toc();

    cout << "Tiempo bucle simple: " << tiempo1 << " segundos\n"
         << "Tiempo bucle paralelizado: " << tiempo2 << " segundos" << endl;

    return 0;
}
