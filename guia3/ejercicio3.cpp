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
    double entrada = -1.5;
    vec membresias = s1.activacionesEntrada(entrada);
    cout << "Grados de activación de los conjuntos trapezoidales para una entrada " << entrada << endl
         << membresias << endl;

    Gnuplot gp;
    mat puntos = join_horiz(ones(membresias.n_elem) * entrada, membresias);
    gp << "set title 'Conjuntos trapezoidales' font ',11'" << endl
       << "set xlabel 'Valor de la variable' font ',10'" << endl
       << "set ylabel 'activacion' font ',10'" << endl
       << "set yrange [0:1.1]" << endl
       << "set key box opaque" << endl
       << "set grid" << endl;
    graficarConjuntos(s1.conjuntosEntrada(), gp);
    gp << gp.file1d(puntos) << "title 'Activaciones' with points ps 1 pt 5 lt rgb 'black'" << endl;
	getchar();

    const mat N = {{0, 1},
                   {1, 2},
                   {3, 2},
                   {5, 1}};

    const SistemaBorroso s2{N, tipoConjunto::gaussiano};
    entrada = 0;
    membresias = s2.activacionesEntrada(entrada);
    cout << "Grados de activación de los conjuntos gaussianos para una entrada " << entrada << endl
         << membresias << endl;

    puntos = join_horiz(ones(membresias.n_elem) * entrada, membresias);
    gp << "set title 'Conjuntos gaussianos'" << endl
       << "set yrange [0:1.1]" << endl
       << "set xrange restore" << endl;
    graficarConjuntos(s2.conjuntosEntrada(), gp);
    gp << gp.file1d(puntos) << "title 'Entrada' with points ps 1 pt 5 lt rgb 'black'" << endl;
    getchar();

	return 0;
}
