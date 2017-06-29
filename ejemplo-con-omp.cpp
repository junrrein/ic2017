#include <armadillo>
#include "gnuplot-iostream.h"

using namespace arma;

int main()
{
    vec A(100000000);

#pragma omp parallel for
    for (unsigned int i = 0; i < A.size(); ++i) {
        A(i) = log(i) * log(i);
    }

    return 0;
}
