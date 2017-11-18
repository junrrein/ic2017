#pragma once

#include <armadillo>
#include "estructura_capas_red.cpp"

using namespace std;
using namespace arma;

namespace ic {

vec sigmoid(const vec& v)
{
	return 2 / (1 + exp(-0.5 * v)) - 1;
}

vector<vec> salidaMulticapa(const vector<mat>& pesos,
                            const vec& patron)
{
	// Este multicapa va a tener salida lineal sólo en la capa de salida.
	// En las demás capas se va a usar salida sigmoidea.
	vector<vec> ySalidas;

	// Calculo de la salida para la primer capa
	{
		vec v = pesos[0] * join_vert(vec{-1}, patron); // agrega entrada correspondiente al sesgo

		if (pesos.size() != 1) { // Si hay más de una capa, a la salida de la primera
			v = sigmoid(v);      // hay que aplicarle sigmoidea.
		}

		ySalidas.push_back(v);
	}

	// Calculo de las salidas para las demas capas
	for (unsigned int i = 1; i < pesos.size(); ++i) {
		vec v = pesos[i] * join_vert(vec{-1}, ySalidas[i - 1]); // agrega entrada correspondiente al sesgo

		if (i != pesos.size() - 1) { // Para todas las capas menos la última, usar salida sigmoidea.
			v = sigmoid(v);
		}

		ySalidas.push_back(v);
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

double errorRelativoPromedioMulticapa(const vector<mat>& pesos,
                                      const mat& patrones,
                                      const mat& salidaDeseada)
{
	double sumaErroresRelativos = 0;

	for (unsigned int n = 0; n < patrones.n_rows; ++n) {
		vector<vec> ySalidas = salidaMulticapa(pesos, patrones.row(n).t());
		const vec salidaRed = ySalidas.back();

		double sumaParcial = 0;

		for (unsigned int i = 0; i < salidaDeseada.n_cols; ++i) {
			const double errorRelativoPatron = abs(salidaRed(i) - salidaDeseada(n, i)) / abs(salidaDeseada(n, i));
			sumaParcial += errorRelativoPatron;
		}

		// Promedio del error absoluto a lo largo de todas las salidas
		// de este patrón.
		sumaErroresRelativos += sumaParcial / salidaDeseada.n_cols;
	}

	// Promedio del error a lo largo de todos los patrones
	return sumaErroresRelativos / patrones.n_rows * 100;
}

vector<mat> epocaMulticapa(const mat& patrones,
                           const mat& salidaDeseada,
                           double tasaAprendizaje,
                           double inercia,
                           vector<mat> pesos)
{
	//Entrenamiento
	vector<mat> deltaWOld;
	for (unsigned int i = 0; i < pesos.size(); ++i)
		deltaWOld.push_back(zeros(pesos[i].n_rows, pesos[i].n_cols));

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
		delta[delta.size() - 1] = error;
		// Deltas de las capas anteriores
		for (int i = ySalidas.size() - 2; i >= 0; --i) {
			// No participan los pesos correspondientes al sesgo en el cálculo de los deltas
			const mat pesosAux = pesos[i + 1].tail_cols(pesos[i + 1].n_cols - 1);
			delta[i] = pesosAux.t() * delta[i + 1];
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

	return pesos;
} // fin funcion Epoca

tuple<vector<mat>, double, int> entrenarMulticapa(const EstructuraCapasRed& estructura,
                                                  const mat& datos,
                                                  int nEpocas,
                                                  double tasaAprendizaje,
                                                  double inercia,
                                                  double tolErrorRelativoPromedio,
                                                  bool monitoreo = false)
{
    mat partEntrenamiento;
    mat partMonitoreo;

    if (monitoreo) {
        const int nEntrenamiento = datos.n_rows * 0.9;
        partEntrenamiento = datos.head_rows(nEntrenamiento);
        partMonitoreo = datos.tail_rows(datos.n_rows - nEntrenamiento);
    }
    else {
        partEntrenamiento = datos;
    }

	const int nSalidas = estructura(estructura.n_elem - 1);
    const int nEntradas = partEntrenamiento.n_cols - nSalidas;
	const int nCapas = estructura.n_elem;
	// Vamos a tener tantas columnas en salidaDeseada
	// como neuronas en la capa de salida
    const mat salidaDeseada = partEntrenamiento.tail_cols(nSalidas);
	// Extender la matriz de patrones con la entrada correspondiente al umbral
    const mat patrones = partEntrenamiento.head_cols(nEntradas);

	// Inicializar pesos y tasa de error
	vector<mat> pesos;

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

	double errorRelativoPromedio;
    double menorErrorMonitoreo = numeric_limits<double>::max();
    vector<mat> mejoresPesos = pesos;
    //    int mejorEpoca = 0;

	// Ciclo de las epocas
	int epoca = 1;
	for (; epoca <= nEpocas; ++epoca) {
		// Ciclo para una época
		pesos = epocaMulticapa(patrones,
		                       salidaDeseada,
		                       tasaAprendizaje,
		                       inercia,
		                       pesos);

		errorRelativoPromedio = errorRelativoPromedioMulticapa(pesos, patrones, salidaDeseada);
        //        cout << epoca << " Error cuadratico entrenamiento: "
        //             << errorCuadraticoMulticapa(pesos, patrones, salidaDeseada) << endl;

        if (monitoreo) {
            const double errorMonitoreo = errorCuadraticoMulticapa(pesos,
                                                                   partMonitoreo.head_cols(nEntradas),
                                                                   partMonitoreo.tail_cols(nSalidas));
            //            cout << "Error cuadratico monitoreo: " << errorMonitoreo << endl;

            if (errorMonitoreo < menorErrorMonitoreo) {
                menorErrorMonitoreo = errorMonitoreo;
                mejoresPesos = pesos;
                //                mejorEpoca = epoca;
            }
        }

        if (errorRelativoPromedio <= tolErrorRelativoPromedio)
            break;
    }
    // Fin ciclo (epocas)

    // Si el bucle anterior no cortó por tolerancia de error,
    // el for va a incrementar la variable una vez de más.
    if (epoca > nEpocas)
        epoca = nEpocas;

    //    cout << "Mejor epoca: " << mejorEpoca << endl;

    if (monitoreo)
        return make_tuple(mejoresPesos, errorRelativoPromedio, epoca);
    else
        return make_tuple(pesos, errorRelativoPromedio, epoca);
}

struct ParametrosMulticapa {
	EstructuraCapasRed estructuraRed;
	int nEpocas;
	double tasaAprendizaje;
	double inercia;
	double toleranciaError;
};
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
