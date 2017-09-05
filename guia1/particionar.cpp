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
		indices = shuffle(indices); // Mezcla las filas de los datos
		particiones.push_back({indices.head_rows(nPatronesEnt),
		                       indices.tail_rows(indices.n_rows - nPatronesEnt)});
	}

	return particiones;
}

void guardarParticiones(const vector<Particion>& particiones, string rutaCarpeta)
{
	int i = 0;

	for (const Particion& particion : particiones) {
		particion.first.save(rutaCarpeta + "particionEnt" + to_string(i));
		particion.second.save(rutaCarpeta + "particionPrueba" + to_string(i));
		++i;
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
