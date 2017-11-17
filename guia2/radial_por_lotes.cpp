#pragma once

#include <armadillo>
#include "estructura_capas_red.cpp"

using namespace std;
using namespace arma;

namespace ic {

enum class tipoInicializacion {
	conjuntosAleatorios,
	patronesAlAzar
};

ivec asignarPatrones(const mat& patrones,
                     vector<rowvec> centroides)
{
	const int nPatrones = patrones.n_rows;
	const int nConjuntos = centroides.size();
	ivec tablaPatronConjunto;
	tablaPatronConjunto.set_size(nPatrones);

	for (int i = 0; i < nPatrones; ++i) {
		vec distancias;
		distancias.set_size(nConjuntos);

		for (int j = 0; j < nConjuntos; ++j) {
			distancias(j) = norm(patrones.row(i) - centroides[j]);
		}

		const int conjuntoGanador = distancias.index_min();
		tablaPatronConjunto(i) = conjuntoGanador;
	}

	return tablaPatronConjunto;
}

// patrones es una matriz que tiene un patrón en cada fila.
// Cada columna representa una dimensión.

// La función devuelve un pair.
// pair.first es un vector de STL que contiene los centroides de cada conjunto.
// pair.second es un vec de Armadillo que contiene el sigma de cada conjunto.
pair<vector<rowvec>, vec>
entrenarRadialPorLotes(const mat& patrones,
                       int nConjuntos,
                       tipoInicializacion tipo = tipoInicializacion::conjuntosAleatorios)
{
	const int nPatrones = patrones.n_rows;

	ivec tablaPatronConjunto;
	tablaPatronConjunto.set_size(nPatrones);
	vector<rowvec> centroides;
	centroides.resize(nConjuntos);

	// 1. Inicialización
	switch (tipo) {
	case tipoInicializacion::conjuntosAleatorios: {
		// Se forman los k conjuntos con patrones elegidos aleatoriamente.
		// Se asigna la misma cantidad de patrones a cada conjunto.
		const uvec indicesMezclados = shuffle(linspace<uvec>(0, nPatrones - 1, nPatrones));
		int conjunto = 0;

		for (int i = 0; i < nPatrones; ++i) {
			// Los patrones no asignados se agregan a cada conjunto,
			// en round-robin.
			tablaPatronConjunto(indicesMezclados(i)) = conjunto;

			if (conjunto + 1 < nConjuntos)
				++conjunto;
			else
				conjunto = 0;
		}

		break;
	}

	case tipoInicializacion::patronesAlAzar: {
		// Se le asignan un patrón aleatorio a cada centroide.
		const uvec indicesMezclados = shuffle(linspace<uvec>(0, nPatrones - 1, nPatrones));
		for (int i = 0; i < nConjuntos; ++i)
			centroides[i] = patrones.row(indicesMezclados(i));

		// Asignar los patrones al conjunto que tiene el centroide mas cercano
		tablaPatronConjunto = asignarPatrones(patrones, centroides);

		break;
	}
	}

	// Bucle del k-medias
	while (true) {
		// 2. Calcular los centroides de cada conjunto
		for (int i = 0; i < nConjuntos; ++i) {
			// Se extraen los indices de los patrones correspondientes al conjunto i
			const uvec indicesConjunto = find(tablaPatronConjunto == i);

			if (!indicesConjunto.empty()) {
				// Se calcula el centroide a lo largo de los patrones del conjunto.
				const rowvec centroide = mean(patrones.rows(indicesConjunto));
				centroides[i] = centroide;
			}
		}

		// 3. Asignar los patrones al conjunto que tiene el centroide mas cercano
		const ivec nuevaTabla = asignarPatrones(patrones, centroides);

		// Detectar si hubo reasignaciones de patrones a conjuntos.
		// Si no hubo reasignaciones, terminamos.
		if (any(tablaPatronConjunto != nuevaTabla))
			tablaPatronConjunto = nuevaTabla;
		else
			break;
	}

	// Terminó el bucle de los k-medias.
	// Ahora hay que calcular el sigma dentro de cada conjunto.
	vec sigmas;
	sigmas.set_size(nConjuntos);

	for (int i = 0; i < nConjuntos; ++i) {
		// Se extraen los indices de los patrones correspondientes al conjunto i
		const uvec indicesConjunto = find(tablaPatronConjunto == i);

		// Se calcula el desvio a lo largo de los patrones del conjunto,
		// a lo largo de las diferentes dimensiones, y luego se saca
		// el promedio de estos desvíos.
		if (!indicesConjunto.empty()) {
			//			const rowvec desvios = stddev(patrones.rows(indicesConjunto));
			//			const double sigma = mean(desvios.t());
			//			sigmas[i] = sigma;

			//            sigmas[i] = 1; // Para el Iris
			//            sigmas[i] = 100; // Para el Merval
            sigmas(i) = 0.7; // Para pronostico con tuplas de 3 series
		}
	}

	// Solo vamos a devolver los centroides y sigmas para los que el conjunto
	// correspondiente no está vacío.
	vector<rowvec> centroidesFinales;
	vec sigmasFinales;

	for (int i = 0; i < nConjuntos; ++i) {
		const uvec indicesConjunto = find(tablaPatronConjunto == i);

		if (!indicesConjunto.empty()) {
			centroidesFinales.push_back(centroides[i]);
			sigmasFinales.insert_rows(sigmasFinales.n_elem, vec{sigmas(i)});
		}
	}

	return {centroidesFinales, sigmasFinales};
}

struct ParametrosRBF {
	EstructuraCapasRed estructuraRed;
	int nEpocas;
	double tasaAprendizaje;
	double inercia;
	double toleranciaError;
};

double gaussiana(const rowvec& patron,
                 const rowvec& centroide,
                 double sigma)
{
	// TODO: Corroborar si hay que elevar la distancia al cuadrado
	return exp(-pow(norm(patron - centroide), 2) / (2 * sigma * sigma));
}

rowvec salidaRadial(const rowvec& patron,
                    vector<rowvec> centroides,
                    vec sigmas)
{
	rowvec salida;
	salida.set_size(centroides.size());

	for (unsigned int i = 0; i < centroides.size(); ++i) {
		salida(i) = gaussiana(patron, centroides[i], sigmas(i));
	}

	return salida;
}

} // namespace ic

istream& operator>>(istream& is, ic::ParametrosRBF& parametros)
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
	if (parametros.estructuraRed.size() != 2
	    || parametros.nEpocas <= 0
	    || parametros.tasaAprendizaje <= 0
	    || parametros.toleranciaError <= 0)
		is.clear(ios::failbit);

	return is;
}
