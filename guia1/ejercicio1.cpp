// http://arma.sourceforge.net/docs.html#example_prog

#include <iostream>
#include <armadillo>
#include <gnuplot-iostream.h>

using namespace std;
using namespace arma;

int main()
{
    arma_rng::set_seed_random();

    mat A = randu<mat>(4, 5);
    mat B = randu<mat>(4, 5);
    mat C = A * B.t();

    cout << C << endl;

    Gnuplot gp;
    gp << "plot " << gp.file1d(vec{C.col(1)}) << "with lines" << endl;

    getchar();

    return 0;
}
