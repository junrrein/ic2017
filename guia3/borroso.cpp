#include <armadillo>
#include <gnuplot-iostream.h>

using namespace arma;
using namespace std;

class Conjunto {

public:
	virtual double membresia(double x) = 0;
};

class ConjuntoTrapezoidal : public Conjunto {
public:
	ConjuntoTrapezoidal(vec puntos);
	double membresia(double x) override;

	const double a, b, c, d;
};

class ConjuntoGaussiano : public Conjunto {
public:
	ConjuntoGaussiano(double t_media, double t_sigma);
	double membresia(double x) override;

	const double media, sigma;
};

ConjuntoTrapezoidal::ConjuntoTrapezoidal(vec puntos)
    : a{puntos(0)}
    , b{puntos(1)}
    , c{puntos(2)}
    , d{puntos(3)}
{
	// chequear orden de puntos
	if (!puntos.is_sorted())
		throw runtime_error("los puntos no estan ordenados");

	if (puntos.n_elem != 4)
		throw runtime_error("no se tienen cuatro puntos");
}

double ConjuntoTrapezoidal::membresia(double x)
{
	if (x < a or x > d)
		return 0;
	else if (a <= x and x < b)
		return (x - a) / (b - a);
	else if (b <= x and x <= c)
		return 1;
	else if (c < x and x <= d)
		return 1 - (x - c) / (d - c);
	else
		throw runtime_error("estamos en la caca");
}

ConjuntoGaussiano::ConjuntoGaussiano(double t_media, double t_sigma)
    : media{t_media}
    , sigma{t_sigma}
{
	if (t_sigma <= 0)
		throw runtime_error("el sigma debe ser mayor o igual a cero");
}

double ConjuntoGaussiano::membresia(double x)
{
	return exp(-0.5 * (pow((x - media) / sigma, 2)));
}

vector<ConjuntoTrapezoidal> parsearTrapecios(const mat& matrizConjuntos)
{
	vector<ConjuntoTrapezoidal> result;

	for (unsigned int i = 0; i < matrizConjuntos.n_rows; ++i) {
		result.push_back(ConjuntoTrapezoidal{matrizConjuntos.row(i).t()});
	}

	return result;
}

vector<ConjuntoGaussiano> parsearGaussianas(const mat& matrizConjuntos)
{
	if (matrizConjuntos.n_cols != 2)
		throw runtime_error("La matriz debe tener dos columnas");

	vector<ConjuntoGaussiano> result;

	for (unsigned int i = 0; i < matrizConjuntos.n_rows; ++i) {
		result.push_back(ConjuntoGaussiano{matrizConjuntos(i, 0),
		                                   matrizConjuntos(i, 1)});
	}

	return result;
}

void graficarTrapecios(const vector<ConjuntoTrapezoidal>& conjuntos, Gnuplot& gp)
{
	gp << "plot ";

	for (unsigned int i = 0; i < conjuntos.size() - 1; ++i) {
		const mat puntos = {{conjuntos[i].a, 0},
		                    {conjuntos[i].b, 1},
		                    {conjuntos[i].c, 1},
		                    {conjuntos[i].d, 0}};

		gp << gp.file1d(puntos) << "with lines title 'Conjunto " << (i + 1) << "', ";
	}

	const mat puntos = {{conjuntos.back().a, 0},
	                    {conjuntos.back().b, 1},
	                    {conjuntos.back().c, 1},
	                    {conjuntos.back().d, 0}};

	gp << gp.file1d(puntos) << "with lines title 'Conjunto " << conjuntos.size() << "'" << endl;
}

void graficarGaussianas(const vector<ConjuntoGaussiano>& conjuntos, Gnuplot& gp)
{
	gp << "plot ";

	for (unsigned int i = 0; i < conjuntos.size() - 1; ++i) {
		const string exponencial = "exp(-0.5 * ((x - "
		                           + to_string(conjuntos[i].media)
		                           + ") / "
		                           + to_string(conjuntos[i].sigma)
		                           + ") ** 2)";

		gp << exponencial << "with lines title 'Conjunto " << (i + 1) << "', ";
	}

	const string exponencial = "exp(-0.5 * ((x - "
	                           + to_string(conjuntos.back().media)
	                           + ") / "
	                           + to_string(conjuntos.back().sigma)
	                           + ") ** 2)";

	gp << exponencial << "with lines title 'Conjunto " << conjuntos.size() << "'" << endl;
}
