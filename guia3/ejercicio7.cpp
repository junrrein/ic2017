#include "borroso.cpp"

int main()
{
    const mat m1 = {{-20, -20, -10, -5},
                    {-10, -5, -5, -2},
                    {-5, -5, -2, 0},
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

    const mat s1 = {{-7, -5, -5, 3},
                    {-5, -3, -3, -1},
                    {-3, -1, -1, 0},
                    {-1, 0, 0, 1},
                    {0, 1, 1, 3},
                    {1, 3, 3, 5},
                    {3, 5, 5, 7}};

    const mat s2 = {{-7, -5, -5, 4},
                    {-5, -4, -4, -3},
                    {-4, -3, -3, 0},
                    {-3, 0, 0, 3},
                    {0, 3, 3, 4},
                    {3, 4, 4, 5},
                    {4, 5, 5, 7}};

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
        q(i) = sistema1.salidaSistema(error(i));
        t_o(i + 1) = t_i + g * q(i) + a * (t_o(i) - t_i);
    }

    Gnuplot gp;
    gp << "set terminal qt size 1100, 600" << endl
       << "set multiplot title 'Evolución del sistema M1 - S1' font ', 12' layout 2, 2" << endl
       // Graficar la salida del sistema
       << "set title 'Salida del sistema' font ', 11'" << endl
       << "set xlabel 'tiempo (segundos)' font ', 10'" << endl
       << "set ylabel 't_o (°C)' font ', 10'" << endl
       << "set yrange [0:26]" << endl
       << "set grid" << endl
       << "plot " << gp.file1d(t_o) << "notitle with lines" << endl
       // Graficar el error del sistema
       << "set title 'Diferencia entre la salida del sistema y la de referencia'" << endl
       << "set ylabel 'error (°C)'" << endl
       << "set yrange restore" << endl
       << "plot " << gp.file1d(error) << "notitle with lines" << endl
       // Graficar la temperatura de referencia del controlador
       << "set title 'Temperatura de referencia'" << endl
       << "set ylabel 't_ref (°C)' font ', 10'" << endl
       << "set yrange [0:26]" << endl
       << "plot " << gp.file1d(t_ref) << "notitle with lines" << endl
       // Graficar la salida del controlador
       << "set title 'Salida del controlador'" << endl
       << "set ylabel 'q'" << endl
       << "set yrange [-7.5:7.5]" << endl
       << "plot " << gp.file1d(q) << "notitle with lines" << endl;
    getchar();

    return 0;
}
