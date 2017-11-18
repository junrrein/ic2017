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

struct ConjuntoDatos {
    mat tuplasEntrada;
    mat tuplasSalida;
};

struct Particion {
    ConjuntoDatos entrenamiento;
    ConjuntoDatos evaluacion;
    ConjuntoDatos prueba;
};

ConjuntoDatos
agruparEntradasConSalidas(const vector<vec>& seriesEntrada,
                          vec serieSalida,
                          unsigned int retrasosEntrada,
                          unsigned int nSalidas)
{
    for (const vec& entrada : seriesEntrada)
        if (entrada.n_elem != serieSalida.n_elem)
            throw runtime_error("Las series de datos deben tener la misma longitud");

    mat tuplasEntrada = agruparEntradas(seriesEntrada, retrasosEntrada, nSalidas);

    // Los primeros retrasosEntrada elementos de serieSalida no pueden usarse como
    // salida deseada, ya que no va a haber retrasosEntrada elementos anteriores
    // para hacer la predicción.
    serieSalida = serieSalida(span(retrasosEntrada, serieSalida.n_elem - 1));
    const mat tuplasSalida = crearTuplas(serieSalida, nSalidas);

    if (tuplasEntrada.n_rows != tuplasSalida.n_rows)
        throw runtime_error("Esto no debería pasar");

    return {tuplasEntrada, tuplasSalida};
}

Particion
armarParticiones(const ConjuntoDatos& datos)
{
    const int nTuplas = datos.tuplasEntrada.n_rows;
    const int nDatosPrueba = nTuplas * 0.1;
    const int nDatosEvaluacion = nTuplas * 0.2;
    const int nDatosEntrenamiento = nTuplas - nDatosPrueba - nDatosEvaluacion;

    Particion particion;
    particion.entrenamiento.tuplasEntrada = datos.tuplasEntrada.head_rows(nDatosEntrenamiento);
    particion.entrenamiento.tuplasSalida = datos.tuplasSalida.head_rows(nDatosEntrenamiento);

    particion.evaluacion.tuplasEntrada = datos.tuplasEntrada.rows(nDatosEntrenamiento,
                                                                  nDatosEntrenamiento + nDatosEvaluacion - 1);
    particion.evaluacion.tuplasSalida = datos.tuplasSalida.rows(nDatosEntrenamiento,
                                                                nDatosEntrenamiento + nDatosEvaluacion - 1);

    particion.prueba.tuplasEntrada = datos.tuplasEntrada.tail_rows(nDatosPrueba);
    particion.prueba.tuplasSalida = datos.tuplasSalida.tail_rows(nDatosPrueba);

    const int N = particion.entrenamiento.tuplasEntrada.n_rows
                  + particion.evaluacion.tuplasEntrada.n_rows
                  + particion.prueba.tuplasEntrada.n_rows;

    if (N != nTuplas)
        throw runtime_error("Esto no tendría que pasar");

    return particion;
}

Particion
cargarTuplas(const vector<string>& rutasSeriesEntrada,
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

    ConjuntoDatos datos = agruparEntradasConSalidas(seriesEntrada,
                                                    serieSalida,
                                                    retrasosEntrada,
                                                    nSalidas);

    return armarParticiones(datos);
}

mat agregarIndiceTemporal(const mat& tuplas)
{
    vec indice = linspace(1, tuplas.n_rows - 1, tuplas.n_rows);
    indice = (indice - min(indice)) / (max(indice) - min(indice));

    return join_horiz(indice, tuplas);
}

//vector<vector<string>>
//subconjuntos(const vector<string>& conjunto)
//{
//    vector<vector<string>> result;

//    for (unsigned int r = 1; r < conjunto.size(); ++r) {
//        vector<string> subconjunto;

//        for (unsigned int i = 0; i < conjunto.size(); ++i) {
//            for (unsigned int j = i; j < i + r; ++j) {
//                subconjunto.push_back(conjunto.at(j));
//            }
//        }

//        result.push_back(subconjunto);
//    }

//    return result;
//}
