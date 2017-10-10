#include "borroso.cpp"

int main()
{
    // Sistema con los conjuntos triangulares originales
	mat M = {{-20, -20, -10, -5},
	         {-10, -5, -5, -2},
	         {-5, -2, -2, 0},
	         {-2, 0, 0, 2},
	         {0, 2, 2, 5},
	         {2, 5, 5, 10},
	         {5, 10, 20, 20}};

	mat S = {{-7, -5, -5, -3},
	         {-5, -3, -3, -1},
	         {-3, -1, -1, 0},
	         {-1, 0, 0, 1},
	         {0, 1, 1, 3},
	         {1, 3, 3, 5},
	         {3, 5, 5, 7}};

	uvec reglas = {1, 2, 3, 4, 5, 6, 7};

	const SistemaBorroso s1{M,
                            tipoConjunto::trapezoidal,
                            S,
                            tipoConjunto::trapezoidal,
                            reglas};

	vec entradas = linspace(-20, 20, 200);
	vec salidas = zeros(200);

	for (unsigned int i = 0; i < entradas.n_elem; ++i)
		salidas(i) = s1.salidaSistema(entradas(i));

	mat puntos = join_horiz(entradas, salidas);

	Gnuplot gp;
    gp << "set terminal qt size 760,760" << endl
       << "set multiplot layout 3, 1" << endl
       << "set title 'Mapeo entrada-salida - Conjuntos triangulares originales' font ', 12'" << endl
       << "set xlabel 'Entrada' font ',11'" << endl
       << "set ylabel 'Salida' font ',11'" << endl
	   << "set xrange [-20:20]" << endl
	   << "set yrange [-7:7]" << endl
       << "set grid" << endl
       << "plot " << gp.file1d(puntos) << "notitle with lines" << endl
       << "set title 'Conjuntos de entrada'" << endl
       << "set xlabel 'x'" << endl
       << "set yrange [0:1.1]" << endl
       << "set key box opaque" << endl;
    s1.graficarConjuntos(Graficar::entrada, gp);
    gp << "NaN notitle" << endl
       << "set title 'Conjuntos de salida'" << endl
       << "set xlabel 'y'" << endl
       << "set xrange [-7:7]" << endl
       << "set yrange [0:1.1]" << endl
       << "set key box opaque" << endl;
    s1.graficarConjuntos(Graficar::salida, gp);
    gp << "NaN notitle" << endl;
	getchar();

    // Sistema con conjuntos trapezoidales
    M = {{-20, -20, -10, -5},
         {-10, -7, -5, -2},
         {-5, -3, -2, 0},
         {-2, -1, 1, 2},
         {0, 2, 3, 5},
         {2, 5, 7, 10},
         {5, 10, 20, 20}};

    S = {{-7, -6, -4, -3},
         {-5, -4, -2, -1},
         {-3, -2, -1, 0},
         {-1, -0.5, 0.5, 1},
         {0, 1, 2, 3},
         {1, 2, 4, 5},
         {3, 4, 6, 7}};

    const SistemaBorroso s2{M,
                            tipoConjunto::trapezoidal,
                            S,
                            tipoConjunto::trapezoidal,
                            reglas};

    entradas = linspace(-20, 20, 200);
    salidas = zeros(200);

    for (unsigned int i = 0; i < entradas.n_elem; ++i)
        salidas(i) = s2.salidaSistema(entradas(i));

    puntos = join_horiz(entradas, salidas);

    Gnuplot gp2;
    gp2 << "set title 'Mapeo entrada-salida - Conjuntos trapezoidales' font ', 12'" << endl
        << "set xlabel 'Entrada' font ',11'" << endl
        << "set ylabel 'Salida' font ',11'" << endl
        << "set xrange [-20:20]" << endl
        << "set yrange [-7:7]" << endl
        << "set grid" << endl
        << "plot " << gp2.file1d(puntos) << "notitle with lines" << endl;
    getchar();

    // Sistema con conjuntos gaussianos
    M = {{-12.5, 3},
         {-6, 2},
         {-2.5, 1},
         {0, 0.5},
         {2.5, 1},
         {6, 2},
         {12.5, 3}};

    S = {{-5, 3},
         {-3, 2},
         {-1.5, 1},
         {0, 0.5},
         {1.5, 1},
         {3, 2},
         {5, 3}};

    const SistemaBorroso s3{M,
                            tipoConjunto::gaussiano,
                            S,
                            tipoConjunto::gaussiano,
                            reglas};

    entradas = linspace(-20, 20, 200);
    salidas = zeros(200);

    for (unsigned int i = 0; i < entradas.n_elem; ++i)
        salidas(i) = s3.salidaSistema(entradas(i));

    puntos = join_horiz(entradas, salidas);

    Gnuplot gp3;
    gp3 << "set title 'Mapeo entrada-salida - Conjuntos gaussianos' font ', 12'" << endl
        << "set xlabel 'Entrada' font ',11'" << endl
        << "set ylabel 'Salida' font ',11'" << endl
        << "set xrange [-20:20]" << endl
        << "set yrange [-7:7]" << endl
        << "set grid" << endl
        << "plot " << gp3.file1d(puntos) << "notitle with lines" << endl;
    getchar();

	return 0;
}
