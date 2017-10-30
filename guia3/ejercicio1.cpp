#include "borroso.cpp"
#include <iostream>

using namespace std;

int main()
{
    ConjuntoTrapezoidal c1{{1, 1, 2, 3}};

    try {
        ConjuntoTrapezoidal c2{{1, 3, 1, 3}};
    }
    catch (runtime_error& e) {
        cerr << "El conjunto c2 no se pudo definir: " << e.what();
    }

    ConjuntoGaussiano c3{0, 1};

    cout << c1.membresia(0) << endl    // tiene que dar cero
         << c1.membresia(1.5) << endl  // tiene que dar uno
         << c1.membresia(2.5) << endl  // deberia dar 0.5
         << c3.membresia(0) << endl    // deberia dar 1
         << c3.membresia(5) << endl    // deberia dar cercano a 0
         << c3.membresia(0.5) << endl; // deberia dar algo entre 0-1

    return 0;
}
