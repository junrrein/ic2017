#include <armadillo>

using namespace std;
using namespace arma;

namespace ic {

enum class tipoInicializacion {
	conjuntosAleatorios,
	valoresAlAzar
};

vec asignarPatrones(const mat& patrones,
                    vector<rowvec> centroides)
{
	const int nPatrones = patrones.n_rows;
	const int nConjuntos = centroides.size();
	vec tablaPatronConjunto;
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

	vec tablaPatronConjunto;
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

	case tipoInicializacion::valoresAlAzar: {
		// Se le asignan valores aleatorios a los centroides.
		// Los valores van a estar en el rango [-0.5; 0.5]
		for (rowvec& centroide : centroides) {
			centroide = randu<rowvec>(patrones.n_cols) - 0.5;
		}

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
		const vec nuevaTabla = asignarPatrones(patrones, centroides);

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
			const rowvec aux = stddev(patrones.rows(indicesConjunto));
			const double sigma = mean(aux);
			sigmas[i] = sigma;
		}
	}

	return {centroides, sigmas};
}

// EstructruraRed especifica la estructura de la red compuesta
// por una capa de base radial mas una capa de salida.
// Por ejemplo, si la red tiene 5 centroides neuronas en la capa de,
// base radial, y el clasificador debe arrojar 3 salidas,
// EstructuraRed para esa red será [5 3].
using EstructuraRed = vec;

struct ParametrosRBF {
	EstructuraRed estructuraRed;
	int nEpocas;
	double tasaAprendizaje;
	double inercia;
	double toleranciaError;
};
}

istream& operator>>(istream& is, ic::EstructuraRed& estructura)
{
	// Formato de estructuraRed:
	// [5 3]
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

	if (estructura.n_elem != 2) // Fallar si se leyó una estructura
		                        // con más o menos de dos capas.
		is.clear(ios::failbit);

	return is;
}

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
	if (parametros.nEpocas <= 0
	    || parametros.tasaAprendizaje <= 0
	    || parametros.toleranciaError <= 0
	    || parametros.toleranciaError >= 100)
		is.clear(ios::failbit);

	return is;
}
