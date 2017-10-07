#include <armadillo>

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

private:
	double a, b, c, d;
};

class ConjuntoGaussiano : public Conjunto {
public:
	ConjuntoGaussiano(double t_media, double t_sigma);
	double membresia(double x) override;

private:
	double media, sigma;
};

ConjuntoTrapezoidal::ConjuntoTrapezoidal(vec puntos)
{
	// chequear orden de puntos
	if (!puntos.is_sorted())
		throw runtime_error("los puntos no estan ordenados");

	if (puntos.n_elem != 4)
		throw runtime_error("no se tienen cuatro puntos");

	a = puntos(0);
	b = puntos(1);
	c = puntos(2);
	d = puntos(3);
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
