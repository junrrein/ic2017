#include <armadillo>
#include <bitset>

using namespace arma;
using namespace std;

mat crearTuplas(vec datos,
                unsigned int longitudTupla)
{
    mat tuplas(datos.n_elem - longitudTupla + 1, longitudTupla);

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

struct VectorParticionado {
    vec entrenamiento;
    vec evaluacion;
    vec prueba;
};

struct ConjuntoDatos {
    mat tuplasEntrada;
    mat tuplasSalida;
};

struct Particion {
    ConjuntoDatos entrenamiento;
    ConjuntoDatos evaluacion;
    ConjuntoDatos prueba;
};

//mat agregarIndiceTemporal(const mat& tuplas)
//{
//    vec indice = linspace(1, tuplas.n_rows - 1, tuplas.n_rows);
//    indice = normalizar(indice, indice);

//    return join_horiz(indice, tuplas);
//}

ConjuntoDatos
agruparEntradasConSalidas(vector<vec> seriesEntrada,
                          vec serieSalida,
                          unsigned int retrasosEntrada,
                          unsigned int nSalidas)
{
    mat tuplasEntrada;
    for (vec& serie : seriesEntrada) {
        // Se eliminan los ultimos nSalidas elementos de cada serie.
        // Como se van a usar como salida deseada, no pueden formar
        // parte de las entradas.
        serie = serie.head(serie.n_elem - nSalidas);
        mat tuplas = crearTuplas(serie, retrasosEntrada);

        tuplasEntrada.insert_cols(tuplasEntrada.n_cols, tuplas);
    }

    // Los primeros retrasosEntrada elementos de serieSalida no pueden usarse como
    // salida deseada, ya que no va a haber retrasosEntrada elementos anteriores
    // para hacer la predicción.
    serieSalida = serieSalida.tail(serieSalida.n_elem - retrasosEntrada);
    const mat tuplasSalida = crearTuplas(serieSalida, nSalidas);

    if (tuplasEntrada.n_rows != tuplasSalida.n_rows)
        throw runtime_error("Esto no debería pasar");

    return {tuplasEntrada, tuplasSalida};
}

Particion
armarTuplas(vector<VectorParticionado> entradasParticionadas,
            VectorParticionado salidaParticionada,
            unsigned int retrasosEntrada,
            unsigned int nSalidas)
{
    vector<vec> seriesEntradaEntrenamiento;
    vector<vec> seriesEntradaEvaluacion;
    vector<vec> seriesEntradaPrueba;
    for (const VectorParticionado& vp : entradasParticionadas) {
        seriesEntradaEntrenamiento.push_back(vp.entrenamiento);
        seriesEntradaEvaluacion.push_back(vp.evaluacion);
        seriesEntradaPrueba.push_back(vp.prueba);
    }

    Particion result;
    result.entrenamiento = agruparEntradasConSalidas(seriesEntradaEntrenamiento,
                                                     salidaParticionada.entrenamiento,
                                                     retrasosEntrada,
                                                     nSalidas);
    result.evaluacion = agruparEntradasConSalidas(seriesEntradaEvaluacion,
                                                  salidaParticionada.evaluacion,
                                                  retrasosEntrada,
                                                  nSalidas);
    result.prueba = agruparEntradasConSalidas(seriesEntradaPrueba,
                                              salidaParticionada.prueba,
                                              retrasosEntrada,
                                              nSalidas);

    return result;
}

pair<vector<VectorParticionado>, VectorParticionado>
particionar(vector<vec> seriesEntrada,
            vec serieSalida)
{
    const int nDatos = serieSalida.n_elem;
    const int nDatosPrueba = nDatos * 0.1;
    const int nDatosEvaluacion = nDatos * 0.2;
    const int nDatosEntrenamiento = nDatos - nDatosPrueba - nDatosEvaluacion;

    vector<VectorParticionado> entradasParticionadas;
    for (const vec& entrada : seriesEntrada) {
        VectorParticionado entradaParticionada;
        entradaParticionada.entrenamiento = entrada.head(nDatosEntrenamiento);
        entradaParticionada.evaluacion = entrada(span(nDatosEntrenamiento,
                                                      nDatosEntrenamiento + nDatosEvaluacion - 1));
        entradaParticionada.prueba = entrada.tail(nDatosPrueba);

        entradasParticionadas.push_back(entradaParticionada);
    }

    VectorParticionado salidaParticionada;
    salidaParticionada.entrenamiento = serieSalida.head(nDatosEntrenamiento);
    salidaParticionada.evaluacion = serieSalida(span(nDatosEntrenamiento,
                                                     nDatosEntrenamiento + nDatosEvaluacion - 1));
    salidaParticionada.prueba = serieSalida.tail(nDatosPrueba);

    return make_pair(entradasParticionadas, salidaParticionada);
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

    for (const vec& serie : seriesEntrada)
        if (serie.n_elem != serieSalida.n_elem)
            throw runtime_error("Las series de datos deben tener la misma longitud");

    // Normalizar todas las series de datos
    for (vec& v : seriesEntrada) {
        v = normalizar(v, v);
    }
    serieSalida = normalizar(serieSalida, serieSalida);

    //    if (agregarIndice) {
    //        vec indice = linspace(1, serieSalida.n_elem - 1, serieSalida.n_elem);
    //        indice = normalizar(indice, indice);

    //        seriesEntrada.push_back(indice);
    //    }

    vector<VectorParticionado> entradasParticionadas;
    VectorParticionado salidaParticionada;
    tie(entradasParticionadas, salidaParticionada) = particionar(seriesEntrada, serieSalida);

    if (salidaParticionada.prueba.n_elem < retrasosEntrada + nSalidas)
        throw runtime_error("No alcanzan los datos de prueba para armar una tupla con la dimensión requerida");

    Particion particion = armarTuplas(entradasParticionadas,
                                      salidaParticionada,
                                      retrasosEntrada,
                                      nSalidas);

    return particion;
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
