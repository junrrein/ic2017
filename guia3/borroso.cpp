#include <armadillo>
#include <gnuplot-iostream.h>
#include <memory>

using namespace arma;
using namespace std;

class Conjunto {
public:
	virtual double membresia(double x) const = 0;
	virtual void graficar(Gnuplot& gp, double escala = 1) const = 0;
	virtual double centroide_x() const = 0;
	virtual double area(double altura) const = 0;
};

class ConjuntoTrapezoidal : public Conjunto {
public:
	ConjuntoTrapezoidal(vec puntos);

	double membresia(double x) const override;
	void graficar(Gnuplot& gp, double escala = 1) const override;
	double centroide_x() const override;
	double area(double altura) const override;

	const double a, b, c, d;
};

class ConjuntoGaussiano : public Conjunto {
public:
	ConjuntoGaussiano(double t_media, double t_sigma);

	double membresia(double x) const override;
	void graficar(Gnuplot& gp, double escala = 1) const override;
	double centroide_x() const override;
	double area(double altura) const override;
	const double media, sigma;
};

enum class tipoConjunto {
	trapezoidal,
	gaussiano
};

enum class Graficar {
	entrada,
	salida
};

class SistemaBorroso {
public:
	SistemaBorroso(const mat& matrizEntrada,
	               tipoConjunto tipoEntrada);
	SistemaBorroso(const mat& matrizEntrada,
	               tipoConjunto tipoEntrada,
	               const mat& matrizSalida,
	               tipoConjunto tipoSalida);

	void graficarConjuntos(Graficar cuales,
	                       Gnuplot& gp,
	                       vec membresias = {}) const;
	vec membresiasEntrada(double x) const;
	double defuzzyficarSalida(vec activaciones) const;

private:
	void graficarConjuntos(const vector<unique_ptr<Conjunto> >& conjuntos,
	                       Gnuplot& gp,
	                       vec escalas = {}) const;

	vector<unique_ptr<Conjunto> > conjuntosEntrada;
	vector<unique_ptr<Conjunto> > conjuntosSalida;
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

double ConjuntoTrapezoidal::membresia(double x) const
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

void ConjuntoTrapezoidal::graficar(Gnuplot& gp, double escala) const
{
	const mat puntos = {{a, 0},
	                    {b, 1 * escala},
	                    {c, 1 * escala},
	                    {d, 0}};

	gp << gp.file1d(puntos) << "with lines";
}

double ConjuntoTrapezoidal::centroide_x() const
{
	// Ver http://www.efunda.com/math/areas/Trapezoid.cfm
	const double A = c - b;
	const double B = d - a;
	const double C = b - a;

	const double numerador = 2 * A * C
	                         + A * A
	                         + C * B
	                         + A * B
	                         + B * B;
	const double denominador = 3 * (A + B);
	const double result = numerador / denominador;

	// El resultado está como si el trapezoide tuviera el punto
	// a en (0, 0).
	// Para corregir esto, hace falta sumarle a.

	return result + a;
}

double ConjuntoTrapezoidal::area(double altura) const
{
	return ((d - a) + (c - b)) * altura / 2;
}

ConjuntoGaussiano::ConjuntoGaussiano(double t_media, double t_sigma)
    : media{t_media}
    , sigma{t_sigma}
{
	if (t_sigma <= 0)
		throw runtime_error("el sigma debe ser mayor o igual a cero");
}

double ConjuntoGaussiano::membresia(double x) const
{
	return exp(-0.5 * (pow((x - media) / sigma, 2)));
}

void ConjuntoGaussiano::graficar(Gnuplot& gp, double escala) const
{
	const string exponencial = to_string(escala)
	                           + " * exp(-0.5 * ((x - "
	                           + to_string(media)
	                           + ") / "
	                           + to_string(sigma)
	                           + ") ** 2)";

	gp << exponencial << "with lines";
}

double ConjuntoGaussiano::centroide_x() const
{
	return media;
}

double ConjuntoGaussiano::area(double altura) const
{
	// PREGUNTAR:
	// ¿Se puede escalar el área original de la gaussiana
	// por el grado de activación y obtener el área correcta?
	return sigma * sqrt(2 * datum::pi) * altura;
}

vector<unique_ptr<Conjunto> > parsearTrapecios(const mat& matrizConjuntos)
{
	vector<unique_ptr<Conjunto> > result;

	for (unsigned int i = 0; i < matrizConjuntos.n_rows; ++i) {
		result.push_back(unique_ptr<Conjunto>{new ConjuntoTrapezoidal{matrizConjuntos.row(i).t()}});
	}

	return result;
}

vector<unique_ptr<Conjunto> > parsearGaussianas(const mat& matrizConjuntos)
{
	if (matrizConjuntos.n_cols != 2)
		throw runtime_error("La matriz debe tener dos columnas");

	vector<unique_ptr<Conjunto> > result;

	for (unsigned int i = 0; i < matrizConjuntos.n_rows; ++i) {
		result.push_back(unique_ptr<Conjunto>{
		    new ConjuntoGaussiano{matrizConjuntos(i, 0),
		                          matrizConjuntos(i, 1)}});
	}

	return result;
}

// NOTA
// Después de usar esta función hace falta cerrar el ploteo
// ya sea ploteando algo más, o ploteando "NaN notitle".
void SistemaBorroso::graficarConjuntos(const vector<unique_ptr<Conjunto> >& conjuntos,
                                       Gnuplot& gp,
                                       vec escalas) const
{
	if (escalas.empty())
		escalas = ones(conjuntos.size());

	gp << "plot ";

	for (unsigned int i = 0; i < conjuntos.size(); ++i) {
		conjuntos[i]->graficar(gp, escalas(i));
		gp << " title 'Conjunto " << (i + 1) << "', ";
	}
}

SistemaBorroso::SistemaBorroso(const mat& matrizEntrada, tipoConjunto tipoEntrada)
{
	switch (tipoEntrada) {
	case tipoConjunto::trapezoidal:
		conjuntosEntrada = parsearTrapecios(matrizEntrada);
		break;

	case tipoConjunto::gaussiano:
		conjuntosEntrada = parsearGaussianas(matrizEntrada);
		break;
	}
}

SistemaBorroso::SistemaBorroso(const mat& matrizEntrada,
                               tipoConjunto tipoEntrada,
                               const mat& matrizSalida,
                               tipoConjunto tipoSalida)
{
	switch (tipoEntrada) {
	case tipoConjunto::trapezoidal:
		conjuntosEntrada = parsearTrapecios(matrizEntrada);
		break;

	case tipoConjunto::gaussiano:
		conjuntosEntrada = parsearGaussianas(matrizEntrada);
		break;
	}

	switch (tipoSalida) {
	case tipoConjunto::trapezoidal:
		conjuntosSalida = parsearTrapecios(matrizSalida);
		break;

	case tipoConjunto::gaussiano:
		conjuntosSalida = parsearGaussianas(matrizSalida);
		break;
	}
}

void SistemaBorroso::graficarConjuntos(Graficar cuales,
                                       Gnuplot& gp,
                                       vec membresias) const
{
	switch (cuales) {
	case (Graficar::entrada):
		graficarConjuntos(conjuntosEntrada, gp, membresias);
		break;

	case (Graficar::salida):
		if (conjuntosSalida.empty())
			throw runtime_error("Este sistema no tiene conjuntos de salida definidos");

		graficarConjuntos(conjuntosSalida, gp, membresias);
		break;
	}
}

vec SistemaBorroso::membresiasEntrada(double x) const
{
	vec result(conjuntosEntrada.size());

	for (unsigned int i = 0; i < conjuntosEntrada.size(); ++i)
		result(i) = conjuntosEntrada[i]->membresia(x);

	return result;
}

double SistemaBorroso::defuzzyficarSalida(vec activaciones) const
{
	if (activaciones.size() != conjuntosSalida.size())
		throw runtime_error(
	      "La cantidad de activaciones y la cantidad "
	      "de conjuntos de salida no coinciden");

	double numerador = 0, denominador = 0;

	for (unsigned int i = 0; i < conjuntosSalida.size(); ++i) {
		numerador += conjuntosSalida[i]->centroide_x()
		             * conjuntosSalida[i]->area(activaciones(i));
		denominador += conjuntosSalida[i]->area(activaciones(i));
	}

	return numerador / denominador;
}
