#include <armadillo>

using namespace arma;
using namespace std;

void crearTuplas(string rutaArchivo,
                 int nEntradas,
                 int nSalidas,
                 string rutaNuevoArchivo)
{

	vec datos;
	datos.load(rutaArchivo);
	const int longitudTupla = nEntradas + nSalidas;

	mat tuplas(datos.n_elem - longitudTupla - 1, longitudTupla);

	for (unsigned int i = 0; i < datos.n_rows - longitudTupla - 1; ++i) {
		tuplas.row(i) = datos.rows(span(i, i + longitudTupla - 1)).t();
	}

	tuplas.save(rutaNuevoArchivo, arma::csv_ascii);
}

void crearTuplas(string rutaArchivo1,
                 string rutaArchivo2,
                 string rutaArchivo3,
                 int nEntradas,
                 int nSalidas,
                 string rutaNuevoArchivo)
{

	vec datos1;
	vec datos2;
	vec datos3;
	datos1.load(rutaArchivo1);
	datos2.load(rutaArchivo2);
	datos3.load(rutaArchivo3);
	const int longitudTupla = nEntradas * 3 + nSalidas;

	mat tuplas(datos1.n_elem - nEntradas - 1, longitudTupla);

	for (unsigned int i = 0; i < datos1.n_rows - nEntradas - nSalidas - 1; ++i) {
		tuplas(i, span(0, tuplas.n_cols - nSalidas - 1))
		    = join_vert(join_vert(datos1.rows(span(i, i + nEntradas - 1)),
		                          datos2.rows(span(i, i + nEntradas - 1))),
		                datos3.rows(span(i, i + nEntradas - 1)))
		          .t();

		tuplas(i,
		       span(tuplas.n_cols - nSalidas,
		            tuplas.n_cols - 1))
		    = datos1(span(i + nEntradas,
		                  i + nEntradas + nSalidas - 1));
	}

	tuplas.save(rutaNuevoArchivo, arma::csv_ascii);
}
