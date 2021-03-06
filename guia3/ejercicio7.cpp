#include "borroso.cpp"
#include <iostream>

void graficosCopados(const SistemaBorroso& s,
                     Gnuplot& gp,
                     const string& titulo);

int main()
{
    const mat m1 = {{-20, -20, -10, -5},
                    {-10, -5, -5, -2},
                    {-5, -2, -2, 0},
                    {-2, 0, 0, 2},
                    {0, 2, 2, 5},
                    {2, 5, 5, 10},
                    {5, 10, 20, 20}};

    const mat m2 = {{-20, -20, -10, -5},
                    {-10, -5, -4, -2},
                    {-4, -2, -1, 0},
                    {-1, 0, 0, 1},
                    {0, 1, 2, 4},
                    {2, 4, 5, 10},
                    {5, 10, 20, 20}};

    const mat m3 = {{-20, -20, -10, -5},
                    {-10, -5, -4, -2},
                    {-4, -2, -0.25, 0},
                    {-0.25, 0, 0, 0.25},
                    {0, 0.25, 2, 4},
                    {2, 4, 5, 10},
                    {5, 10, 20, 20}};

    const mat s1 = {{-7, -5, -5, -3},
                    {-5, -3, -3, -1},
                    {-3, -1, -1, 0},
                    {-1, 0, 0, 1},
                    {0, 1, 1, 3},
                    {1, 3, 3, 5},
                    {3, 5, 5, 7}};

    const mat s2 = {{-7, -5, -5, -4},
                    {-5, -4, -4, -3},
                    {-4, -3, -3, 0},
                    {-3, 0, 0, 3},
                    {0, 3, 3, 4},
                    {3, 4, 4, 5},
                    {4, 5, 5, 7}};

    const mat s3 = {{-10, -7, -7, -4},
                    {-6, -4, -4, -2},
                    {-1.4, -0.4, -0.4, 0.6},
                    {-0.25, 0, 0, 0.25},
                    {-0.6, 0.4, 0.4, 1.4},
                    {2, 4, 4, 6},
                    {4, 7, 7, 10}};

    const uvec reglas = {1, 2, 3, 4, 5, 6, 7};

    const SistemaBorroso sistema1{m1,
                                  tipoConjunto::trapezoidal,
                                  s1,
                                  tipoConjunto::trapezoidal,
                                  reglas};
    const SistemaBorroso sistema2{m1,
                                  tipoConjunto::trapezoidal,
                                  s2,
                                  tipoConjunto::trapezoidal,
                                  reglas};
    const SistemaBorroso sistema3{m2,
                                  tipoConjunto::trapezoidal,
                                  s1,
                                  tipoConjunto::trapezoidal,
                                  reglas};
    const SistemaBorroso sistema4{m2,
                                  tipoConjunto::trapezoidal,
                                  s2,
                                  tipoConjunto::trapezoidal,
                                  reglas};
    const SistemaBorroso sistema5{m3,
                                  tipoConjunto::trapezoidal,
                                  s3,
                                  tipoConjunto::trapezoidal,
                                  reglas};

    Gnuplot gp1;
    graficosCopados(sistema1, gp1, "M1 - S1");
    getchar();

    Gnuplot gp2;
    graficosCopados(sistema2, gp2, "M1 - S2");
    getchar();

    Gnuplot gp3;
    graficosCopados(sistema3, gp3, "M2 - S1");
    getchar();

    Gnuplot gp4;
    graficosCopados(sistema4, gp4, "M2 - S2");
    getchar();

    Gnuplot gp5;
    graficosCopados(sistema5, gp5, "M2 - S3");
    getchar();

    return 0;
}

void graficosCopados(const SistemaBorroso& s,
                     Gnuplot& gp,
                     const string& titulo)
{
    const double t_i = 15;
    const double g = 40.0 / 41, a = g;
    const vec t_ref = join_vert(ones(30) * 15, ones(170) * 25);
    const double t_o_inicial = 15;
    // IMPORTANTE
    // to va a tener base 1.
    // Es decir, el primer valor del vector (indice 0) corresponde
    // al instante -1.
    vec t_o = join_vert(vec{t_o_inicial}, zeros(200));
    vec error = zeros(200);
    vec q = zeros(200);

    for (unsigned int i = 0; i < t_ref.n_elem; ++i) {
        error(i) = t_ref(i) - t_o(i);
        q(i) = s.salidaSistema(error(i));
        t_o(i + 1) = t_i + g * q(i) + a * (t_o(i) - t_i);
    }

    // ------------------------
    // Demasiados gráficos
    // ------------------------
    //    gp << "set terminal qt size 1800, 900" << endl
    //       << "set multiplot title 'Evolución del sistema " + titulo + "' font ', 12' layout 2, 2" << endl

    //       // Graficar el error del sistema
    //       << "set title 'Diferencia entre la salida del sistema y la de referencia' font ',11'" << endl
    //       << "set xlabel 'Tiempo (segundos)' font ',10'" << endl
    //       << "set ylabel 'error (°C)'" << endl
    //       << "set xrange [0:199]" << endl
    //       << "set yrange [-20:20]" << endl
    //       << "set grid" << endl
    //       << "plot " << gp.file1d(error) << "notitle with lines" << endl

    //       // Graficar los conjuntos de entrada del controlador
    //       << "set title 'Conjuntos de entrada'" << endl
    //       << "set xlabel 'error (°C)' font ', 10'" << endl
    //       << "set ylabel ''" << endl
    //       << "set xrange [-20:20]" << endl
    //       << "set yrange [0:1.1]" << endl;
    //    graficarConjuntos(s.conjuntosEntrada(), gp);
    //    gp << "NaN notitle" << endl;

    //    // Graficar la salida del controlador
    //    gp << "set title 'Salida del controlador'" << endl
    //       << "set xlabel 'Tiempo (segundos)'" << endl
    //       << "set ylabel 'q'" << endl
    //       << "set xrange [0:199]" << endl
    //       << "set yrange [-7:7]" << endl
    //       << "plot " << gp.file1d(q) << "notitle with lines" << endl

    //       // Graficar los conjuntos de salida del controlador
    //       << "set title 'Conjuntos de salida'" << endl
    //       << "set xlabel 'q'" << endl
    //       << "set ylabel ''" << endl
    //       << "set xrange [-7:7]" << endl
    //       << "set yrange [0:1.1]" << endl;
    //    graficarConjuntos(s.conjuntosSalida(), gp);
    //    gp << "NaN notitle" << endl;

    gp << "set title 'Evolución del sistema " + titulo + "' font ', 11'" << endl
       << "set xlabel 'tiempo (segundos)' font ', 10'" << endl
       << "set ylabel 't (°C)' font ', 10'" << endl
       << "set yrange [14:26]" << endl
       << "set grid" << endl
       << "set key box opaque" << endl
       << "plot " << gp.file1d(t_o(span(1, t_o.n_elem - 1)).eval()) << "title 'Salida del sistema' with lines lw 3, "
       << gp.file1d(t_ref) << "title 'Salida de referencia' with lines lw 0.5" << endl;

    cout << titulo << endl
         << "Error de estado estacionario: " << error(error.n_elem - 1) << endl
         << endl;
}
