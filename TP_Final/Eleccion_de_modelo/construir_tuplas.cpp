#include <armadillo>
#include <bitset>

using namespace arma;
using namespace std;

mat crearTuplas(vec datos,
                unsigned int longitudTupla)
{
    mat tuplas(datos.n_elem - longitudTupla, longitudTupla);

    for (unsigned int i = 0; i < tuplas.n_rows; ++i) {
        rowvec tupla = datos(span(i, i + longitudTupla - 1)).t();
        tuplas.row(i) = tupla;
    }

    return tuplas;
}

vec normalizar(vec serieOriginal, vec serieACambiar)
{
    const double minimo = min(serieOriginal);
    const double rango = max(serieOriginal) - minimo;

    vec result = (serieACambiar - minimo) / rango;

    return result;
}

vec desnormalizar(vec serieOriginal, vec serieNormalizada)
{
    const double minimo = min(serieOriginal);
    const double rango = max(serieOriginal) - minimo;

    vec result = serieNormalizada * rango + minimo;

    return result;
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

mat agregarIndiceTemporal(const mat& tuplas)
{
    vec indice = linspace(1, tuplas.n_rows - 1, tuplas.n_rows);
    indice = normalizar(indice, indice);

    return join_horiz(indice, tuplas);
}

ConjuntoDatos
agruparEntradasConSalidas(vector<vec> seriesEntrada,
                          vec serieSalida,
                          unsigned int retrasosEntrada,
                          unsigned int nSalidas,
                          bool agregarIndice = false)
{
    for (const vec& entrada : seriesEntrada)
        if (entrada.n_elem != serieSalida.n_elem)
            throw runtime_error("Las series de datos deben tener la misma longitud");

    // Normalizar todas las series de datos
    for (vec& v : seriesEntrada) {
        v = normalizar(v, v);
    }
    serieSalida = normalizar(serieSalida, serieSalida);

    mat tuplasEntrada = agruparEntradas(seriesEntrada, retrasosEntrada, nSalidas);

    // Los primeros retrasosEntrada elementos de serieSalida no pueden usarse como
    // salida deseada, ya que no va a haber retrasosEntrada elementos anteriores
    // para hacer la predicción.
    serieSalida = serieSalida(span(retrasosEntrada, serieSalida.n_elem - 1));
    const mat tuplasSalida = crearTuplas(serieSalida, nSalidas);

    if (tuplasEntrada.n_rows != tuplasSalida.n_rows)
        throw runtime_error("Esto no debería pasar");

    if (agregarIndice)
        tuplasEntrada = agregarIndiceTemporal(tuplasEntrada);

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
             unsigned int nSalidas,
             bool agregarIndice = false)
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
                                                    nSalidas,
                                                    agregarIndice);

    return armarParticiones(datos);
}

template <unsigned int N>
vector<vector<string>>
subconjuntos(const vector<string>& conjunto)
{
    const int nElemResult = pow(2, conjunto.size());
    vector<vector<string>> result;

    if (N != conjunto.size())
        throw runtime_error("Pone bien los parámetros cacho");

    //    https://www.quora.com/How-do-I-generate-all-subsets-of-a-set-in-C++-iteratively
    bitset<N> setDeBits{0};

    for (int i = 0; i < nElemResult; ++i) {
        vector<string> subconjunto;

        for (unsigned int k = 0; k < conjunto.size(); k++) {
            if (setDeBits.test(k)) {
                subconjunto.push_back(conjunto.at(k));
            }
        }

        result.push_back(subconjunto);
        setDeBits = bitset<N>{setDeBits.to_ulong() + 1};
    }

    return result;
}
