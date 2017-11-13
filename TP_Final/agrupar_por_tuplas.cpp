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
