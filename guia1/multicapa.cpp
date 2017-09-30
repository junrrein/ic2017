#pragma once

#include <armadillo>

using namespace std;
using namespace arma;

namespace ic {
// EstructruraCapaRed especifica la cantidad de neuronas en cada capa.
// Por ejemplo, si la red tiene 2 neuronas en la primera capa,
// 3 en la capa oculta y 1 en la capa de salida,
// EstructuraCapasRed para esa red será [2 3 1].
using EstructuraCapasRed = vec;

vec sigmoid(const vec& v)
{
	return 2 / (1 + exp(-v)) - 1;
}

vec winnerTakesAll(const vec& v)
{
	if (v.n_elem == 1)
		// Si la red tiene una sola salida, aplicar la función signo
		return v(0) >= 0 ? vec{1} : vec{-1};
	else {
		vec result = v;
		// Si la red tiene varias salidas, asignarle +1 a "la que ganó"
		// y -1 a las demás.
		const double maximo = max(result);

		result.transform([maximo](double val) {
            // Comparación de igualdad de números de coma flotante
            if (abs(val - maximo) < 1e-9)
                return 1;
            else
                return -1;
		});

		return result;
	}
}

vector<vec> salidaMulticapa(const vector<mat>& pesos,
                            const vec& patron)
{
	vector<vec> ySalidas;

	// Calculo de la salida para la primer capa
	{
		const vec v = pesos[0] * join_vert(vec{-1}, patron); // agrega entrada correspondiente al sesgo
		ySalidas.push_back(sigmoid(v));
	}

	// Calculo de las salidas para las demas capas
	for (unsigned int i = 1; i < pesos.size(); ++i) {
		const vec v = pesos[i] * join_vert(vec{-1}, ySalidas[i - 1]); // agrega entrada correspondiente al sesgo
		ySalidas.push_back(sigmoid(v));                               // TODO: ver qué valor le pasamos como parámetro
		                                                              // a la sigmoidea
	}

	return ySalidas;
}

double errorCuadraticoMulticapa(const vector<mat>& pesos,
                                const mat& patrones,
                                const mat& salidaDeseada)
{
	double errorCuadraticoTotal = 0;

	for (unsigned int n = 0; n < patrones.n_rows; ++n) {
		vector<vec> ySalidas = salidaMulticapa(pesos, patrones.row(n).t());
		const vec salidaRed = ySalidas.back();

		const double errorCuadraticoPatron = sum(pow(salidaDeseada.row(n).t() - salidaRed, 2));
		errorCuadraticoTotal += errorCuadraticoPatron;
	}

	return errorCuadraticoTotal;
}

double errorClasificacionMulticapa(const vector<mat>& pesos,
                                   const mat& patrones,
                                   const mat& salidaDeseada)
{
	int errores = 0;

	for (unsigned int n = 0; n < patrones.n_rows; ++n) {
		vector<vec> ySalidas = salidaMulticapa(pesos, patrones.row(n).t());

		const vec salidaRed = winnerTakesAll(ySalidas.back()); // fija los valores en -1 o 1.

		if (any(salidaRed.t() != salidaDeseada.row(n)))
			++errores;
	}

	double tasaError = static_cast<double>(errores) / patrones.n_rows * 100;

	return tasaError;
}

double errorClasificacionMulticapa(const vector<mat>& pesos,
                                   const mat& datos)
{
	const int nEntradas = pesos.front().n_cols - 1;
	const int nSalidas = pesos.back().n_rows;
	if (nEntradas + nSalidas != int(datos.n_cols))
		throw runtime_error("Están mal calculados el número de entradas y salidas");

	return errorClasificacionMulticapa(pesos,
	                                   datos.head_cols(nEntradas),
	                                   datos.tail_cols(nSalidas));
}

pair<vector<mat>, double> epocaMulticapa(const mat& patrones,
                                         const mat& salidaDeseada,
                                         double tasaAprendizaje,
                                         double inercia,
                                         vector<mat> pesos)
{
	//Entrenamiento
	vector<mat> deltaWOld;
	for (unsigned int i = 0; i < pesos.size(); ++i)
		deltaWOld.push_back(zeros(size(pesos[i])));

	for (unsigned int n = 0; n < patrones.n_rows; ++n) {
		// Calcular las salidas para cada capa
		const vector<vec> ySalidas = salidaMulticapa(pesos, patrones.row(n).t());

		// Calculo del error
		const vec error = salidaDeseada.row(n).t() - ySalidas.back();

		// Calculo retropropagacion
		// Calculo de gradiente error local instantaneo
		vector<vec> delta;
		// Tenemos tantos vectores de deltas como capas
		delta.resize(ySalidas.size());

		// Delta de ultima capa
		delta[delta.size() - 1] = error
		                          % (1 + ySalidas.back())
		                          % (1 - ySalidas.back());
		// Deltas de las capas anteriores
		for (int i = ySalidas.size() - 2; i >= 0; --i) {
			// No participan los pesos correspondientes al sesgo en el cálculo de los deltas
			const mat pesosAux = pesos[i + 1].tail_cols(pesos[i + 1].n_cols - 1);
			delta[i] = (pesosAux.t() * delta[i + 1])
			           % (1 + ySalidas[i])
			           % (1 - ySalidas[i]);
		}

        // Actualizacion de pesos de todas las capas menos la primera.
        // Si la red tiene una sola capa, no se entra acá.
		for (int i = pesos.size() - 1; i >= 1; --i) {
			const mat deltaWnuevo = tasaAprendizaje
			                            * delta[i]
			                            * join_horiz(vec{-1}, ySalidas[i - 1].t())
			                        + inercia * deltaWOld[i];
			pesos[i] += deltaWnuevo;
			deltaWOld[i] = deltaWnuevo;
		}

		// Actualización de pesos de la primer capa
		const mat deltaW = tasaAprendizaje
		                       * delta[0]
		                       * join_horiz(vec{-1}, patrones.row(n))
		                   + inercia * deltaWOld[0];
		pesos[0] += deltaW;
		deltaWOld[0] = deltaW;
	}

	// Calculo de Tasa de error
	double tasaError = errorClasificacionMulticapa(pesos, patrones, salidaDeseada);

	return {pesos, tasaError};
} // fin funcion Epoca

tuple<vector<mat>, vec, vec, int> entrenarMulticapa(const EstructuraCapasRed& estructura,
                                                    const mat& datos,
                                                    int nEpocas,
                                                    double tasaAprendizaje,
                                                    double inercia,
                                                    double toleranciaErrorClasificacion,
                                                    long long semilla = -1)
{
	const int nSalidas = estructura(estructura.n_elem - 1);
	const int nEntradas = datos.n_cols - nSalidas;
	const int nCapas = estructura.n_elem;
	// Vamos a tener tantas columnas en salidaDeseada
	// como neuronas en la capa de salida
	const mat salidaDeseada = datos.tail_cols(nSalidas);
	// Extender la matriz de patrones con la entrada correspondiente al umbral
	const mat patrones = datos.head_cols(nEntradas);

	// Inicializar pesos y tasa de error
	vector<mat> pesos;

#pragma omp critical(inicializarPesos)
	{
		if (semilla != -1)
			arma_rng::set_seed(semilla);

		// La primer matriz matriz de pesos tiene tantas filas como neuronas en la primer capa
		// y tantas columnas como componentes tiene la entrada, más la entrada correspondiente
		// al sesgo.
		pesos.push_back(randu(estructura(0), nEntradas + 1) - 0.5);

		for (int i = 1; i < nCapas; ++i) {
			// Las siguientes matrices de pesos tienen tantas filas como neuronas en dicha capa
			// y tantas columnas como entradas a esa capa, que van a ser las salidas de
			// la capa anterior mas la entrada correspondiente al sesgo.
			// Las salidas de la capa anterior es igual al nro de neuronas en la capa anterior.
			pesos.push_back(randu(estructura(i), estructura(i - 1) + 1) - 0.5);
		}
	}

	double tasaError = 100;
	vec erroresClasificacion;
	vec erroresCuadraticos;

	// Ciclo de las epocas
	int epoca = 1;
	for (; epoca <= nEpocas; ++epoca) {
		// Ciclo para una época
		tie(pesos, tasaError) = epocaMulticapa(patrones,
		                                       salidaDeseada,
		                                       tasaAprendizaje,
		                                       inercia,
		                                       pesos);

		erroresClasificacion.insert_rows(erroresClasificacion.n_elem, vec{tasaError});
		const double errorCuadratico = errorCuadraticoMulticapa(pesos, patrones, salidaDeseada);
		erroresCuadraticos.insert_rows(epoca - 1, vec{errorCuadratico});

		if (tasaError <= toleranciaErrorClasificacion)
			break;
	}
	// Fin ciclo (epocas)

	// Si el bucle anterior no cortó por tolerancia de error,
	// el for va a incrementar la variable una vez de más.
	if (epoca > nEpocas)
		epoca = nEpocas;

	return make_tuple(pesos, erroresClasificacion, erroresCuadraticos, epoca);
}

struct ParametrosMulticapa {
	EstructuraCapasRed estructuraRed;
	int nEpocas;
	double tasaAprendizaje;
	double inercia;
	double toleranciaError;
};
}

istream& operator>>(istream& is, ic::EstructuraCapasRed& estructura)
{
	// Formato de estructuraCapasRed:
	// [2 3 1]
	{
		char ch;
		is >> ch;
		if (ch != '[') {            // Si lo leído no empieza con corchete, la estructura
			is.clear(ios::failbit); // está mal formateada.
			return is;
		}
	}

	// Asegurarnos de que la variable 'estructura' esté vacía
	estructura.clear();

	// Primero chequear si estamos por leer un número (con el primer caracter)
	// y después leerlo posta.
	for (char ch; is >> ch;) {
		if (isdigit(ch)) {
			is.unget();
			int numero;
			is >> numero;

			if (numero == 0) { // No podemos tener una capa con cero neuronas
				is.clear(ios::failbit);
				return is;
			}

			estructura.insert_rows(estructura.n_elem, vec{double(numero)});
		}
		else if (ch == ']') { // Cuando se encuentra el corchete que cierra,
			break;            // se terminó de leer la estructura.
		}
		else {                      // Si lo que se leyó no es número ni corchete, la estructura leída
			is.clear(ios::failbit); // tiene formato erróneo.
			return is;
		}
	}

	if (estructura.empty()) // Fallar si lo que se leyó es "[]"
		is.clear(ios::failbit);

	return is;
}

istream& operator>>(istream& is, ic::ParametrosMulticapa& parametros)
{
	// Formato:
	// estructura: [3 2 1]
	// n_epocas: 200
	// tasa_entrenamiento: 0.1
	// inercia: 0.5
	// parametro_sigmoidea: 1
	// tolerancia_error: 5
	string str;

	// No chequeamos si la etiqueta de cada línea está bien o no. No nos importa
	is >> str >> parametros.estructuraRed
	    >> str >> parametros.nEpocas
	    >> str >> parametros.tasaAprendizaje
	    >> str >> parametros.inercia
	    >> str >> parametros.toleranciaError;

	// Control básico de valores de parámetros
	if (parametros.nEpocas <= 0
	    || parametros.tasaAprendizaje <= 0
	    || parametros.toleranciaError <= 0
	    || parametros.toleranciaError >= 100)
		is.clear(ios::failbit);

	return is;
}
