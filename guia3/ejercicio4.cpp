#include "borroso.cpp"

int main()
{
    const mat S1 = {{-7, -5, -5, -3},
                    {-5, -3, -3, -1},
                    {-3, -1, -1, 0},
                    {-1, 0, 0, 1},
                    {0, 1, 1, 3},
                    {1, 3, 3, 5},
                    {3, 5, 5, 7}};

    const SistemaBorroso s1{S1,
                            tipoConjunto::trapezoidal,
                            S1,
                            tipoConjunto::trapezoidal};

    const vec activaciones = {0, 0.7, 0.3, 0, 0, 0, 0};
    double salida = s1.defuzzyficarSalida(activaciones);
    cout << "Salida defuzzyficada: " << salida << endl;

    Gnuplot gp;
    mat puntoSalida = {{salida, 0}};
    gp << "set terminal qt size 600,600" << endl
       << "set multiplot layout 2,1" << endl
       << "set title 'Conjuntos de salida' font ',11'" << endl
       << "set xlabel 'y' font ', 10'" << endl
       << "set yrange [0:1.1]" << endl
       << "set key box opaque" << endl
       << "set grid" << endl;
    graficarConjuntos(s1.conjuntosSalida(), gp);
    gp << "NaN notitle" << endl
       << "set title 'Defuzzyficacion de la salida' font ', 11'" << endl
       << "set ylabel 'activacion' font ', 10'" << endl
       << "set grid" << endl
       << "set key box opaque" << endl;
    graficarConjuntos(s1.conjuntosSalida(), gp, activaciones);
    gp << gp.file1d(puntoSalida) << "title 'Salida final' with points ps 2 pt 5 lt rgb 'black'" << endl
       << "unset multiplot" << endl;
    getchar();

    const mat S2 = {{-5, 2},
                    {-3, 1},
                    {-1, 0.8},
                    {0, 0.5},
                    {1, 0.8},
                    {3, 1},
                    {5, 2}};

    const SistemaBorroso s2{S2,
                            tipoConjunto::gaussiano,
                            S2,
                            tipoConjunto::gaussiano};

    salida = s2.defuzzyficarSalida(activaciones);
    cout << "Salida defuzzyficada: " << salida << endl;

    puntoSalida = {{salida, 0}};
    gp << "set multiplot layout 2,1" << endl
       << "set title 'Conjuntos de salida' font ',11'" << endl
       << "set xlabel 'y' font ', 10'" << endl
       << "set yrange [0:1.1]" << endl
       << "set key box opaque" << endl
       << "set grid" << endl;
    graficarConjuntos(s2.conjuntosSalida(), gp);
    gp << "NaN notitle" << endl
       << "set title 'Defuzzyficacion de la salida' font ', 11'" << endl
       << "set ylabel 'activacion' font ', 10'" << endl
       << "set xrange restore" << endl
       << "set grid" << endl
       << "set key box opaque" << endl;
    graficarConjuntos(s2.conjuntosSalida(), gp, activaciones);
    gp << gp.file1d(puntoSalida) << "title 'Salida final' with points ps 2 pt 5 lt rgb 'black'" << endl
       << "unset multiplot" << endl;
    getchar();

	return 0;
}
