#include <armadillo>
#include <vector>
#include <set>
#include <boost/algorithm/cxx11/any_of.hpp>

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
    vector<int> camino;
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
    nodoInicial.camino.push_back(nodoInicial.ciudadFinal);
    nodoInicial.costoCamino = 0;

    listaCaminos.insert(nodoInicial);
}

vector<int> ArbolBusqueda::hacerBusqueda()
{
    while (!listaCaminos.empty()) {
        // bucle:
        //  1 - Saco el primer elemento de la lista
        //      Si no hay elementos, no encontré una solución
        Camino camino = *listaCaminos.begin();
        listaCaminos.erase(listaCaminos.begin());

        //  2 - Me fijo si es un estado objetivo
        //      Si es un estado objetivo, encontré la solución
        // El camino va a ser la solución si la cantidad de nodos es igual
        // a la cantidad de ciudades.
        if (camino.camino.size() == cantidadCiudades)
            return camino.camino;

        //  3 - Expando los vecinos a los que puede ir y los meto en ListaNodos Arbol
        //      Los nuevos Nodos son insertados para que lista quede ordenada por
        //      el costo de cada camino
        //      Si no hay nodos a expandir, no inserto nada
        vector<int> ciudadesCandidatas;
        for (unsigned int i = 1; i <= cantidadCiudades; ++i) {
            if (!boost::algorithm::any_of_equal(camino.camino, i))
                ciudadesCandidatas.push_back(i);
        }

        // Generar caminos nuevos
        for (int ciudad : ciudadesCandidatas) {
            Camino nuevoCamino{camino};
            nuevoCamino.ciudadFinal = ciudad;
            nuevoCamino.camino.push_back(ciudad);
            nuevoCamino.costoCamino += distancias.at(camino.ciudadFinal,
                                                     nuevoCamino.ciudadFinal);
        }
    }

    // Si se llegó hasta acá, es porque no se encontró una solución
    throw runtime_error("No se encontró una solución");
}
