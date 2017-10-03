#pragma once

#include <armadillo>

using namespace arma;
using namespace std;

// EstructruraCapaRed especifica la cantidad de neuronas en cada capa.
// Por ejemplo, si la red tiene 2 neuronas en la primera capa,
// 3 en la capa oculta y 1 en la capa de salida,
// EstructuraCapasRed para esa red será [2 3 1].

namespace ic {

using EstructuraCapasRed = vec;
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
