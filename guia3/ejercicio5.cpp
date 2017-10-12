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
    const mat puntoEntrada = {{entrada, 0}};
    const mat puntoSalida = {{salida, 0}};

    Gnuplot gp;
    gp << "set terminal qt size 700,700" << endl
       << "set multiplot layout 2, 1" << endl
       << "set title 'Fuzzyficacion de la entrada' font ', 12'" << endl
       << "set xlabel 'x' font ', 11'" << endl
       << "set ylabel 'activacion' font ', 11'" << endl
       << "set xrange [-21:21]" << endl
       << "set yrange [0:1.1]" << endl
       << "set grid" << endl
       << "set key box opaque" << endl;
    graficarConjuntos(s.conjuntosEntrada(), gp, actEntrada);
    gp << gp.file1d(puntoEntrada) << "title 'Entrada al sistema' with points ps 2 lt rgb 'blue'" << endl;

    gp << "set title 'Defuzzyficacion de la salida' font ', 12'" << endl
       << "set xlabel 'y' font ', 11'" << endl
       << "set ylabel 'activacion' font ', 11'" << endl
       << "set xrange [-8:8]" << endl
       << "set yrange [0:1.1]" << endl
       << "set grid" << endl
       << "set key box opaque" << endl;
    graficarConjuntos(s.conjuntosSalida(), gp, actSalida);
    gp << gp.file1d(puntoSalida) << "title 'Salida del sistema' with points ps 2 lt rgb 'blue'" << endl;

    getchar();

    return 0;
}
