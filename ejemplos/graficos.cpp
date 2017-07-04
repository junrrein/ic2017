#include <armadillo>
#include "../gnuplot-iostream.h"

using namespace arma;

int main()
{
    vec x = linspace(0, 2 * M_PI, 50);
    vec y = sin(x);
    mat puntos = join_horiz(x, y);

    Gnuplot gp;

    // Ejemplo básico
    gp << "plot " << gp.file1d(y) << "with lines" << endl;

    cout << "Presionar Enter para continuar\n";
    getchar();

    // "plot" de Matlab
    gp << "set title '\"Plot\" de Matlab' font ',13'\n"
       << "set xlabel 'Tiempo (segundos)'\n"
       << "set ylabel 'Magnitud'\n"
       << "set xrange [0 : 2*pi]\n"
       << "set grid\n"
       << "unset key\n" // Ocultar la leyenda
       << "plot " << gp.file1d(puntos) << "with lines" << endl;

    cout << "Presionar Enter para continuar\n";
    getchar();

    // "stem" de Matlab
    gp << "set title '\"Stem\" de Matlab' font ',13'\n"
       << "plot " << gp.file1d(puntos) << "with impulses, "
       << gp.file1d(puntos) << "with points pt 7 lt 1" << endl;

    cout << "Presionar Enter para continuar\n";
    getchar();

    // "subplot" de Matlab
    gp << "set terminal qt size 600,650\n"
       << "set multiplot layout 2, 1 title '\"Subplot\" de Matlab' font ',14'\n"
       << "set title '\"Plot\" de Matlab' font ',13'\n"
       << "plot " << gp.file1d(puntos) << "with lines\n"
       << "set title '\"Stem\" de Matlab' font ',13'\n"
       << "plot " << gp.file1d(puntos) << "with impulses, "
       << gp.file1d(puntos) << "with points pt 7 lt 1\n"
       << "unset multiplot" << endl;

    cout << "Presionar Enter para continuar\n";
    getchar();

    return 0;
}
