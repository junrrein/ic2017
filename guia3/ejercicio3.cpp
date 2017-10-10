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
    const mat N = {{0, 1},
                   {1, 2},
                   {3, 2},
                   {5, 1}};

    SistemaBorroso s1{N, tipoConjunto::gaussiano};
    const vec membresias = s1.membresiasEntrada(0);
    cout << membresias << endl;

    Gnuplot gp;
    s1.graficarConjuntos(Graficar::entrada, gp, membresias);
    gp << "NaN notitle" << endl;
	getchar();

	return 0;
}
