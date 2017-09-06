#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

namespace ic {
using Particion = pair<uvec, uvec>;

vector<Particion> particionar(mat datos, int nParticiones, double porcentajeEnt)
{
	vector<Particion> particiones;
	const int nPatronesEnt = datos.n_rows * porcentajeEnt / 100;
	uvec indices = linspace<uvec>(0, datos.n_rows - 1, datos.n_rows);

	for (int i = 0; i < nParticiones; ++i) {
		indices = shuffle(indices);
		particiones.push_back({indices.head_rows(nPatronesEnt),
		                       indices.tail_rows(indices.n_rows - nPatronesEnt)});
	}

	return particiones;
}

vector<Particion> leaveKOut(mat datos, int k)
{
	vector<Particion> particiones;
	const int nParticiones = datos.n_rows / k; // TODO: Preguntar qué hacer cuando la cantidad
	                                           // de patrones no es divisible por k
	const uvec indices = shuffle(linspace<uvec>(0, datos.n_rows - 1, datos.n_rows));

	for (int i = 0; i < nParticiones; ++i) {
		const uvec indicesPrueba = indices.rows(span(k * i, k * i + k - 1));
		uvec indicesEnt;

		if (k * i - 1 >= 0)
			indicesEnt.insert_rows(0, indices.rows(span(0, k * i - 1)));

		if (k * i + k <= int(indices.n_elem - 1))
			indicesEnt.insert_rows(0, indices.rows(span(k * i + k, indices.n_elem - 1)));

		particiones.push_back({indicesEnt, indicesPrueba});
	}

	return particiones;
}

void guardarParticiones(const vector<Particion>& particiones, string rutaCarpeta)
{
	for (unsigned int i = 0; i < particiones.size(); ++i) {
		particiones[i].first.save(rutaCarpeta + "particionEnt" + to_string(i), arma_ascii);
		particiones[i].second.save(rutaCarpeta + "particionPrueba" + to_string(i), arma_ascii);
	}
}

vector<Particion> cargarParticiones(string rutaCarpeta, int nParticiones)
{
	vector<Particion> particiones;
	particiones.resize(nParticiones);

	for (int i = 0; i < nParticiones; ++i) {
		particiones[i].first.load(rutaCarpeta + "particionEnt" + to_string(i));
		particiones[i].second.load(rutaCarpeta + "particionPrueba" + to_string(i));
	}

	return particiones;
}
}
