#include <armadillo>

using namespace arma;
using namespace std;

mat crearTuplas(vec datos,
                unsigned int N)
{
    datos = (datos - min(datos)) / (max(datos) - min(datos));

    mat tuplas(datos.n_elem - N, N);

    for (unsigned int i = 0; i < tuplas.n_rows; ++i) {
        rowvec tupla = datos(span(i, i + N - 1)).t();
        tuplas.row(i) = tupla;
    }

    return tuplas;
}

mat agruparEntradas(unsigned int nSalidas,
                    vec serieDatos,
                    unsigned int nEntradas)
{
    serieDatos = serieDatos(span(0, serieDatos.n_elem - 1 - nSalidas));

    return crearTuplas(serieDatos, nEntradas);
}

template <typename... Args>
mat agruparEntradas(unsigned int nSalidas,
                    vec serieDatos,
                    unsigned int nEntradas,
                    Args... args)
{
    serieDatos = serieDatos(span(0, serieDatos.n_elem - 1 - nSalidas));

    return join_horiz(crearTuplas(serieDatos, nEntradas),
                      agruparEntradas(nSalidas, args...));
}

//template <typename... Args>
//mat agruparEntradasConSalidas(string rutaSalida,
//                              unsigned int nSalidas,
//                              Args... args)
//{
//}
