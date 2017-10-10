#include "borroso.cpp"

int main()
{
    const mat M = {{-20, -20, -10, -5},
                   {-10, -5, -5, -2},
                   {-5, -2, -2, 0},
                   {-2, 0, 0, 2},
                   {0, 2, 2, 5},
                   {2, 5, 5, 10},
                   {5, 10, 20, 20}};
    const SistemaBorroso s1{M, tipoConjunto::trapezoidal};

    Gnuplot gp;

    gp << "set yrange[0:1.2]" << endl
       << "set key box opaque width 3" << endl;
    s1.graficarConjuntos(Graficar::entrada, gp);
    gp << "NaN notitle" << endl;
    getchar();

    const mat N = {{0, 1},
                   {1, 2},
                   {3, 2},
                   {5, 1}};
    const SistemaBorroso s2{N, tipoConjunto::gaussiano};

    s2.graficarConjuntos(Graficar::entrada, gp);
    getchar();

    return 0;
}
