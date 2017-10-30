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

    const mat S = {{-7, -5, -5, -3},
                   {-5, -3, -3, -1},
                   {-3, -1, -1, 0},
                   {-1, 0, 0, 1},
                   {0, 1, 1, 3},
                   {1, 3, 3, 5},
                   {3, 5, 5, 7}};

    const uvec reglas = {1, 2, 3, 4, 5, 6, 7};

    const SistemaBorroso s{M,
                           tipoConjunto::trapezoidal,
                           S,
                           tipoConjunto::trapezoidal,
                           reglas};

    const double entrada = 6;
    const vec actEntrada = s.activacionesEntrada(entrada);
    const vec actSalida = s.mapeoEntradaSalida(actEntrada);
    const double salida = s.defuzzyficarSalida(actSalida);
    const mat puntoSalida = {{salida, 0}};

    Gnuplot gp;
    mat puntosEntrada = join_horiz(ones(actEntrada.n_elem) * entrada, actEntrada);
    gp << "set terminal qt size 600,600" << endl
       << "set multiplot layout 2, 1" << endl
       << "set title 'Fuzzyficacion de la entrada' font ', 11'" << endl
       << "set xlabel 'x' font ', 10'" << endl
       << "set ylabel 'activacion' font ', 10'" << endl
       << "set yrange [0:1.1]" << endl
       << "set grid" << endl
       << "set key box opaque top left" << endl;
    graficarConjuntos(s.conjuntosEntrada(), gp);
    gp << gp.file1d(puntosEntrada) << "title 'Activaciones' with points ps 2 lt rgb 'blue'" << endl;

    gp << "set title 'Defuzzyficacion de la salida'" << endl
       << "set xlabel 'y'" << endl
       << "set ylabel 'activacion'" << endl;
    graficarConjuntos(s.conjuntosSalida(), gp, actSalida);
    gp << gp.file1d(puntoSalida) << "title 'Salida del sistema' with points ps 2 lt rgb 'blue'" << endl;

    getchar();

    return 0;
}
