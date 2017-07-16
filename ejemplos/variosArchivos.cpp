#include "funcion1.hpp"
#include "funcion2.hpp"
#include <armadillo>
#include <gnuplot-iostream.h>

using namespace arma;

int main()
{
    hacerAlgo();
    hacerOtraCosa();

    vec datos;
    datos.load("sin.txt");

    Gnuplot gp;
    gp << "plot " << gp.file1d(datos) << "with lines" << std::endl;

    getchar();

    return 0;
}
