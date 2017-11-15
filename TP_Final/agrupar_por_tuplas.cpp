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

	datos = (datos - min(datos)) / (max(datos) - min(datos));

	const int longitudTupla = nEntradas + nSalidas;

	// FIXME: Cuando la cantidad de salidas es mayor a 1
	// no se calcula bien la cantidad de tuplas
	mat tuplas(datos.n_elem - nEntradas, longitudTupla);

	for (unsigned int i = 0; i < tuplas.n_rows; ++i) {
		const rowvec tupla = datos(span(i, i + longitudTupla - 1)).t();
		tuplas.row(i) = tupla;
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

	//	mat datos1;
	vec datos1;
	vec datos2;
	vec datos3;
	datos1.load(rutaArchivo1);
	datos2.load(rutaArchivo2);
	datos3.load(rutaArchivo3);

	//	const vec meses = datos1.col(0);
	//	vec datos1 = datos1.col(1);

	// Normalizar datos de exportaciones e importacioes
	datos1 = (datos1 - min(datos1)) / (max(datos1) - min(datos1));
	datos2 = (datos2 - min(datos2)) / (max(datos2) - min(datos2));
	datos3 = (datos3 - min(datos3)) / (max(datos3) - min(datos3));

	const int longitudTupla = nEntradas * 3 + nSalidas;

	mat tuplas(datos1.n_elem - nEntradas - nSalidas - 1, longitudTupla);

	for (unsigned int i = 0; i < tuplas.n_rows; ++i) {
		tuplas(i, span(0, tuplas.n_cols - nSalidas - 1))
		    = join_vert(join_vert(datos1.rows(span(i, i + nEntradas - 1)),
		                          datos2.rows(span(i, i + nEntradas - 1))),
		                datos3.rows(span(i, i + nEntradas - 1)))
		          .t();

		tuplas(i,
		       span(tuplas.n_cols - nSalidas,
		            tuplas.n_cols - 1))
		    = datos1(span(i + nEntradas,
		                  i + nEntradas + nSalidas - 1))
		          .t();
	}

	tuplas.save(rutaNuevoArchivo, arma::csv_ascii);
}
