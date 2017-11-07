#include <armadillo>
#include <vector>
#include <set>
#include <boost/algorithm/cxx11/none_of.hpp>

using namespace arma;
using namespace std;

// matriz distancias

// nodoArbolInicial = rand()
// ListaNodosArbol = {nodoArbolInicial}

// Un NodoArbol sabe:
//  * El camino recorrido (vector<int>)
//  * El costo de ese camino
struct Camino {
    int ciudadFinal;
    vector<int> recorrido;
    double costoCamino;
};

bool operator<(const Camino& lhs, const Camino& rhs)
{
    return lhs.costoCamino < rhs.costoCamino;
}

class ArbolBusqueda {
public:
    ArbolBusqueda(string rutaArchivoDistancias);

    vector<int> hacerBusqueda();

private:
    set<Camino> listaCaminos;
    unsigned int cantidadCiudades;
    mat distancias;
};

ArbolBusqueda::ArbolBusqueda(string rutaArchivoDistancias)
{
    distancias.load(rutaArchivoDistancias);

    if (!distancias.is_square())
        throw runtime_error("La matriz leída no es cuadrada");

    cantidadCiudades = distancias.n_rows;

    Camino nodoInicial;
    nodoInicial.ciudadFinal = randi(1, distr_param(1, cantidadCiudades)).at(0);
    nodoInicial.recorrido.push_back(nodoInicial.ciudadFinal);
    nodoInicial.costoCamino = 0;

    listaCaminos.insert(nodoInicial);
}

vector<int> ArbolBusqueda::hacerBusqueda()
{
    while (!listaCaminos.empty()) {
        //  1 - Saco el primer elemento de la lista
        //      Si no hay elementos, no encontré una solución
        Camino camino = *listaCaminos.begin();
        listaCaminos.erase(listaCaminos.begin());

        //  2 - Me fijo si es un estado objetivo
        //      Si es un estado objetivo, encontré la solución
        // El camino va a ser la solución si la cantidad de nodos es igual
        // a la cantidad de ciudades.
        if (camino.recorrido.size() == cantidadCiudades)
            return camino.recorrido;

        //  3 - Expando los vecinos a los que puede ir y los meto en listaCaminos.
        //      Los nuevos Caminos son insertados para que lista quede ordenada por
        //      el costo de cada camino.
        //      Si no hay vecinos a expandir, no inserto nada.
        for (unsigned int ciudad = 1; ciudad <= cantidadCiudades; ++ciudad) {

            // Si la ciudad no está en el recorrido hecho, entonces puedo viajar
            // a esa ciudad.
            if (boost::algorithm::none_of_equal(camino.recorrido, ciudad)) {

                // Creo un nuevo camino en base al actual
                Camino nuevoCamino{camino};

                // Actualizo el nuevo camino con la información sobre la
                // nueva ciudad a la que se viajó.
                nuevoCamino.ciudadFinal = ciudad;
                nuevoCamino.recorrido.push_back(ciudad);
                nuevoCamino.costoCamino += distancias.at(camino.ciudadFinal - 1,
                                                         nuevoCamino.ciudadFinal - 1);

                // Inserto el nuevo camino
                listaCaminos.insert(nuevoCamino);
            }
        }
    }

    // Si se llegó hasta acá, es porque no se encontró una solución
    throw runtime_error("No se encontró una solución");
}
