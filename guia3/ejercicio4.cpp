#include "borroso.cpp"

int main()
{
	const mat N = {{0, 1},
	               {1, 2},
	               {3, 2},
	               {5, 1}};
    const mat S = {{-7, -5, -5, -3},
                   {-5, -3, -3, -1},
                   {-3, -1, -1, 0},
                   {-1, 0, 0, 1},
                   {0, 1, 1, 3},
                   {1, 3, 3, 5},
                   {3, 5, 5, 7}};

    SistemaBorroso s2{N,
                      tipoConjunto::gaussiano,
                      S,
                      tipoConjunto::trapezoidal};

    const vec activaciones = {0, 0.7, 0.3, 0, 0, 0, 0};
    const double salida = s2.defuzzyficarSalida(activaciones);
    cout << "Salida defuzzyficada: " << salida << endl;

    Gnuplot gp;
    const mat puntoSalida = {{salida, 0}};
    gp << "set title 'Defuzzyficacion de la salida' font ', 12'" << endl
       << "set xlabel 'y' font ', 11'" << endl
       << "set ylabel 'activacion' font ', 11'" << endl
       << "set xrange [-8:8]" << endl
       << "set yrange [0:1.1]" << endl
       << "set grid" << endl
       << "set key box opaque width auto" << endl;
    s2.graficarConjuntos(Graficar::salida, gp, activaciones);
    gp << gp.file1d(puntoSalida) << "title 'Salida final' with points ps 2 lt rgb 'blue'" << endl;
    getchar();

	return 0;
}
