#include <armadillo>

using namespace arma;
using namespace std;

mat crearTuplas(vec datos,
                unsigned int longitudTupla)
{
    datos = (datos - min(datos)) / (max(datos) - min(datos));

    mat tuplas(datos.n_elem - longitudTupla, longitudTupla);

    for (unsigned int i = 0; i < tuplas.n_rows; ++i) {
        rowvec tupla = datos(span(i, i + longitudTupla - 1)).t();
        tuplas.row(i) = tupla;
    }

    return tuplas;
}

mat agruparEntradas(vector<vec> seriesDatos,
                    unsigned int retrasosEntrada,
                    unsigned int nSalidas)
{
    // Se asume que todas las series de datos tienen
    // la misma longitud.
    mat result;

    for (vec& serie : seriesDatos) {
        // Se eliminan los ultimos nSalidas elementos de cada serie.
        // Como se van a usar como salida deseada, no pueden formar
        // parte de las entradas.
        serie = serie(span(0, serie.n_elem - 1 - nSalidas));
        mat tuplas = crearTuplas(serie, retrasosEntrada);

        result.insert_cols(result.n_cols, tuplas);
    }

    return result;
}

pair<mat, mat> agruparEntradasConSalidas(const vector<vec>& seriesEntrada,
                                         vec serieSalida,
                                         unsigned int retrasosEntrada,
                                         unsigned int nSalidas)
{
    for (const vec& entrada : seriesEntrada)
        if (entrada.n_elem != serieSalida.n_elem)
            throw runtime_error("Las series de datos deben tener la misma longitud");

    mat entradasTuplas = agruparEntradas(seriesEntrada, retrasosEntrada, nSalidas);

    // Los primeros retrasosEntrada elementos de serieSalida no pueden usarse como
    // salida deseada, ya que no va a haber retrasosEntrada elementos anteriores
    // para hacer la predicción.
    serieSalida = serieSalida(span(retrasosEntrada, serieSalida.n_elem - 1));
    const mat salidaTuplas = crearTuplas(serieSalida, nSalidas);

    if (entradasTuplas.n_rows != salidaTuplas.n_rows)
        throw runtime_error("Esto no debería pasar");

    return make_pair(entradasTuplas, salidaTuplas);
}

pair<mat, mat> cargarTuplas(const vector<string>& rutasSeriesEntrada,
                            const string& rutaSerieSalida,
                            unsigned int retrasosEntrada,
                            unsigned int nSalidas)
{
    vector<vec> seriesEntrada;

    for (const string& ruta : rutasSeriesEntrada) {
        seriesEntrada.push_back(vec{});
        seriesEntrada.back().load(ruta);
    }

    vec serieSalida;
    serieSalida.load(rutaSerieSalida);

    return agruparEntradasConSalidas(seriesEntrada,
                                     serieSalida,
                                     retrasosEntrada,
                                     nSalidas);
}

mat agregarIndiceTemporal(const mat& tuplas)
{
    vec indice = linspace(1, tuplas.n_rows - 1, tuplas.n_rows);
    indice = (indice - min(indice)) / (max(indice) - min(indice));

    return join_horiz(indice, tuplas);
}
