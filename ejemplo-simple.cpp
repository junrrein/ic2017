#include <armadillo>
#include "gnuplot-iostream.h"

using namespace arma;

int main()
{
    vec x = linspace(0, 2 * M_PI, 200);
    vec y = sin(x);
    mat points = join_horiz(x, y);

    Gnuplot gp;
    gp << "set title 'sin(x)' font ',13'\n"
       << "set xlabel 'Tiempo (segundos)'\n"
       << "set ylabel 'Magnitud'\n"
       << "set xrange [0 : 2*pi]\n"
       << "set grid\n"
       << "set nokey\n" // Ocultar la leyenda
       << "plot " << gp.file1d(points) << "with lines" << endl;

    getchar();

    return 0;
}
