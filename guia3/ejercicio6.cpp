#include "borroso.cpp"

void graficoLoco(const SistemaBorroso& s,
                 Gnuplot& gp,
                 const string& titulo,
                 pair<double, double> limitesX,
                 pair<double, double> limitesY);

int main()
{
    // --------------------------------------------------
    // Sistema con los conjuntos triangulares originales
    // --------------------------------------------------
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

    Gnuplot gp1;
    graficoLoco(s1,
                gp1,
                "Conjuntos triangulares originales",
                {-20, 20},
                {-7, 7});
    getchar();

    // ------------------------------------
    // Sistema con conjuntos trapezoidales
    // ------------------------------------
    M = {{-20, -20, -15, -7},
         {-15, -7, -5, -3},
         {-5, -3, -2, -1},
         {-2, -1, 1, 2},
         {1, 2, 3, 5},
         {3, 5, 7, 8},
         {7, 8, 20, 20}};

    S = {{-7, -6, -5, -4},
         {-5, -4, -3, -2},
         {-3, -2, -1, -0.5},
         {-1, -0.5, 0.5, 1},
         {0.5, 1, 2, 3},
         {2, 3, 4, 5},
         {4, 5, 6, 7}};

    const SistemaBorroso s2{M,
                            tipoConjunto::trapezoidal,
                            S,
                            tipoConjunto::trapezoidal,
                            reglas};

    Gnuplot gp2;
    graficoLoco(s2,
                gp2,
                "Conjuntos trapezoidales",
                {-20, 20},
                {-7, 7});
    getchar();

    // ---------------------------------
    // Sistema con conjuntos gaussianos
    // ---------------------------------
    M = {{0, 1},
         {4, 1},
         {8, 1}};

    S = {{0, 0.5},
         {1.5, 0.25},
         {3, 0.5}};

    const SistemaBorroso s3{M,
                            tipoConjunto::gaussiano,
                            S,
                            tipoConjunto::gaussiano,
                            {1, 2, 3}};

    Gnuplot gp3;
    graficoLoco(s3,
                gp3,
                "Conjuntos gaussianos",
                {0, 10},
                {0, 5});
    getchar();

    // ---------------------------------------
    // Sistema con conjuntos trapezoidales
    // Usando mapeo entre conjuntos invertido
    // ---------------------------------------
    M = {{-20, -20, -15, -7},
         {-15, -7, -5, -3},
         {-5, -3, -2, -1},
         {-2, -1, 1, 2},
         {1, 2, 3, 5},
         {3, 5, 7, 8},
         {7, 8, 20, 20}};

    S = {{-7, -6, -5, -4},
         {-5, -4, -3, -2},
         {-3, -2, -1, -0.5},
         {-1, -0.5, 0.5, 1},
         {0.5, 1, 2, 3},
         {2, 3, 4, 5},
         {4, 5, 6, 7}};

    const SistemaBorroso s4{M,
                            tipoConjunto::trapezoidal,
                            S,
                            tipoConjunto::trapezoidal,
                            {7, 6, 5, 4, 3, 2, 1}};
    Gnuplot gp4;
    graficoLoco(s4,
                gp4,
                "Conjuntos trapezoidales - Mapeo inverso",
                {-20, 20},
                {-7, 7});
    getchar();

	return 0;
}

void graficoLoco(const SistemaBorroso& s,
                 Gnuplot& gp,
                 const string& titulo,
                 pair<double, double> limitesX,
                 pair<double, double> limitesY)
{
    const vec entradas = linspace(limitesX.first, limitesX.second, 200);
    vec salidas = zeros(200);

    for (unsigned int i = 0; i < entradas.n_elem; ++i)
        salidas(i) = s.salidaSistema(entradas(i));

    const mat puntos = join_horiz(entradas, salidas);
    const string rangoEntrada = "[" + to_string(limitesX.first) + ":" + to_string(limitesX.second) + "]";
    const string rangoSalida = "[" + to_string(limitesY.first) + ":" + to_string(limitesY.second) + "]";

    gp << "set terminal qt size 550,680" << endl
       << "set multiplot layout 3, 1" << endl
       << "set title 'Mapeo entrada-salida - " + titulo + "' font ', 12'" << endl
       << "set xlabel 'Entrada' font ',11'" << endl
       << "set ylabel 'Salida' font ',11'" << endl
       << "set xrange " + rangoEntrada << endl
       << "set yrange " + rangoSalida << endl
       << "set grid" << endl
       << "plot " << gp.file1d(puntos) << "notitle with lines" << endl
       << "set title 'Conjuntos de entrada'" << endl
       << "set xlabel 'x'" << endl
       << "unset ylabel" << endl
       << "set yrange [0:1.1]" << endl
       << "set key box opaque" << endl;
    graficarConjuntos(s.conjuntosEntrada(), gp);
    gp << "NaN notitle" << endl
       << "set title 'Conjuntos de salida'" << endl
       << "set xlabel 'y'" << endl
       << "set xrange " + rangoSalida << endl
       << "set yrange [0:1.1]" << endl
       << "set key box opaque" << endl;
    graficarConjuntos(s.conjuntosSalida(), gp);
    gp << "NaN notitle" << endl;
}
