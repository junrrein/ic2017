#include <armadillo>

using namespace arma;
using namespace std;

void crearTuplas(string rutaArchivo,
                 int nEntradas,
                 int nSalidas,
                 string rutaNuevoArchivo)
{
    mat datos;
    datos.load(rutaArchivo);

    vec meses = datos.col(0);
    vec ventas = datos.col(1);

    meses = (meses - min(meses)) / (max(meses) - min(meses));
    ventas = (ventas - min(ventas)) / (max(ventas) - min(ventas));

    const int longitudTupla = 1 + nEntradas + nSalidas;

	// FIXME: Cuando la cantidad de salidas es mayor a 1
	// no se calcula bien la cantidad de tuplas
    mat tuplas(ventas.n_elem - nEntradas - nSalidas, longitudTupla);

	for (unsigned int i = 0; i < tuplas.n_rows; ++i) {
        const rowvec tupla = join_horiz(rowvec{meses(i)},
                                        ventas(span(i, i + longitudTupla - 2)).t());
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

	mat datos1;
	vec datos2;
	vec datos3;
	datos1.load(rutaArchivo1);
	datos2.load(rutaArchivo2);
	datos3.load(rutaArchivo3);

    vec meses = datos1.col(0);
	vec ventas = datos1.col(1);

    // Normalizar datos de exportaciones e importacioes
    meses = (meses - min(meses)) / (max(meses) - min(meses));
	ventas = (ventas - min(ventas)) / (max(ventas) - min(ventas));
	datos2 = (datos2 - min(datos2)) / (max(datos2) - min(datos2));
	datos3 = (datos3 - min(datos3)) / (max(datos3) - min(datos3));

	const int longitudTupla = 1 + nEntradas * 3 + nSalidas;

	mat tuplas(ventas.n_elem - nEntradas - nSalidas - 1, longitudTupla);

	for (unsigned int i = 0; i < tuplas.n_rows; ++i) {
		tuplas(i, 0) = meses(i);

		tuplas(i, span(1, tuplas.n_cols - nSalidas - 1))
		    = join_vert(join_vert(ventas.rows(span(i, i + nEntradas - 1)),
		                          datos2.rows(span(i, i + nEntradas - 1))),
		                datos3.rows(span(i, i + nEntradas - 1)))
		          .t();

		tuplas(i,
		       span(tuplas.n_cols - nSalidas,
		            tuplas.n_cols - 1))
		    = ventas(span(i + nEntradas,
		                  i + nEntradas + nSalidas - 1))
		          .t();
	}

	tuplas.save(rutaNuevoArchivo, arma::csv_ascii);
}
