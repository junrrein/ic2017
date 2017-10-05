#include <armadillo>
#include <gnuplot-iostream.h>

using namespace arma;
using namespace std;

namespace ic {

class SOM {
public:
    // Constructores
    SOM(const mat& patrones, pair<int, int> dimensiones, const vec& salidaDeseada = {});

    // Interfaz
    void entrenar(int nEpocas,
                  double velocidadInicial,
                  double velocidadFinal,
                  int vecindadInicial,
                  int vecindadFinal);
    void etiquetar();
    int clasificar(const rowvec& patron) const;
    vec clasificar(const mat& patrones) const;
    void graficar(Gnuplot& gp, bool graficarVecindades = true) const;

    // Acceso a miembros
    const field<rowvec>& mapa() const { return m_mapa; };
    const mat& etiquetas() const { return m_etiquetas; };
    const mat& patrones() const { return m_patrones; };

private:
    field<rowvec> m_mapa;
    mat m_etiquetas;
    const mat m_patrones;
    const vec m_salidaDeseada;

    pair<pair<int, int>, double> buscarGanadora(const rowvec& patron) const;
};

SOM::SOM(const mat& patrones, pair<int, int> dimensiones, const vec& salidaDeseada)
    : m_patrones{patrones}
    , m_salidaDeseada{salidaDeseada}
{
    m_mapa = field<rowvec>(dimensiones.first, dimensiones.second);

    // Inicializar mapa
    for (unsigned int i = 0; i < m_mapa.n_rows; ++i) {
        for (unsigned int j = 0; j < m_mapa.n_cols; ++j) {
            m_mapa(i, j) = randu<rowvec>(patrones.n_cols) - 0.5;
        }
    }
}

pair<pair<int, int>, double> SOM::buscarGanadora(const rowvec& patron) const
{
    pair<int, int> coordGanadora;
    double distanciaGanadora = numeric_limits<double>::max();

    for (unsigned int j = 0; j < m_mapa.n_rows; ++j) {
        for (unsigned int k = 0; k < m_mapa.n_cols; ++k) {
            const double distancia = norm(patron - m_mapa(j, k));

            if (distancia < distanciaGanadora) {
                distanciaGanadora = distancia;
                coordGanadora = {j, k};
            }
        }
    }

    return {coordGanadora, distanciaGanadora};
}

void SOM::entrenar(int nEpocas,
                   double velocidadInicial,
                   double velocidadFinal,
                   int vecindadInicial,
                   int vecindadFinal)
{
    const vec velocidad = linspace(velocidadInicial, velocidadFinal, nEpocas);
    // Redondeamos la vecindad porque necesitamos que tenga valores enteros
    const vec vecindad = round(linspace(vecindadInicial, vecindadFinal, nEpocas));

    for (int epoca = 0; epoca < nEpocas; ++epoca) {
        for (unsigned int n = 0; n < m_patrones.n_rows; ++n) {

            // Buscamos la neurona ganadora
            pair<int, int> coordGanadora;
            double distanciaGanadora;
            tie(coordGanadora, distanciaGanadora) = buscarGanadora(m_patrones.row(n));

            // Adaptación de pesos
            const int maxX = int(m_mapa.n_rows - 1);
            const int maxY = int(m_mapa.n_cols - 1);
            const int xInicial = (coordGanadora.first - vecindad(epoca) < 0) ? 0 : (coordGanadora.first - vecindad(epoca));
            const int xFinal = (coordGanadora.first + vecindad(epoca) > maxX) ? maxX : (coordGanadora.first + vecindad(epoca));
            const int yInicial = (coordGanadora.second - vecindad(epoca) < 0) ? 0 : (coordGanadora.second - vecindad(epoca));
            const int yFinal = (coordGanadora.second + vecindad(epoca) > maxY) ? maxY : (coordGanadora.second + vecindad(epoca));

            for (int x = xInicial; x <= xFinal; ++x) {
                for (int y = yInicial; y <= yFinal; ++y) {
                    m_mapa(x, y) += velocidad(epoca) * (m_patrones.row(n) - m_mapa(x, y));
                }
            }
        }
    }
}

void SOM::etiquetar()
{
    if (m_salidaDeseada.empty())
        throw runtime_error("Este SOM no posee una salida deseada asociada a los patrones");

    field<ivec> mapaContador(m_mapa.n_rows, m_mapa.n_cols);

    // Inicializar los contadores de clases
    for (unsigned int x = 0; x < m_mapa.n_rows; ++x) {
        for (unsigned int y = 0; y < m_mapa.n_cols; ++y) {
            mapaContador(x, y) = zeros<ivec>(2);
        }
    }

    // Contamos, para cada neurona, qué cantidad de veces gana para cada clase
    for (unsigned int n = 0; n < m_patrones.n_rows; ++n) {
        pair<int, int> ganadora;
        tie(ganadora, ignore) = buscarGanadora(m_patrones.row(n));

        if (m_salidaDeseada(n) == 0)
            mapaContador(ganadora.first, ganadora.second).at(0) += 1;
        else
            mapaContador(ganadora.first, ganadora.second).at(1) += 1;
    }

    m_etiquetas = mat(m_mapa.n_rows, m_mapa.n_cols);

    // Se asignan las etiquetas de clase
    for (unsigned int x = 0; x < m_mapa.n_rows; ++x) {
        for (unsigned int y = 0; y < m_mapa.n_cols; ++y) {
            if (mapaContador(x, y)(0) == mapaContador(x, y)(1))
                // Si la neurona ganó la misma cantidad de veces para ambas clases,
                // se le asigna la clase al azar.
                m_etiquetas(x, y) = as_scalar(randi(1, distr_param(0, 1)));
            else
                m_etiquetas(x, y) = index_max(mapaContador(x, y));
        }
    }
}

int SOM::clasificar(const rowvec& patron) const
{
    pair<int, int> ganadora;
    tie(ganadora, ignore) = buscarGanadora(patron);

    return m_etiquetas(ganadora.first, ganadora.second);
}

vec SOM::clasificar(const mat& patrones) const
{
    vec result = zeros(patrones.n_rows);

    for (unsigned int n = 0; n < patrones.n_rows; ++n) {
        result(n) = clasificar(rowvec{patrones.row(n)});
    }

    return result;
}

void SOM::graficar(Gnuplot& gp, bool graficarVecindades) const
{
    gp << "set key box opaque width 3" << endl
       << "set xlabel 'x_1' font ',11'" << endl
       << "set ylabel 'x_2' font ',11'" << endl

       // Graficar patrones
       << "plot " << gp.file1d(m_patrones) << "title 'Patrones' with points pt 2 ps 1 lt rgb 'blue', ";

    // Graficar neuronas del mapa y las conexiones
    for (unsigned int x = 0; x < m_mapa.n_rows; ++x) {
        for (unsigned int y = 0; y < m_mapa.n_cols; ++y) {
            // Graficar la neurona
            gp << gp.file1d(m_mapa(x, y).eval()) << "notitle with points ps 2 pt 1 lt -1 lw 3, ";

            if (graficarVecindades) {
                // Graficar conexiones con las vecinas horizontales y verticales
                if (x != 0)
                    gp << gp.file1d(join_vert(m_mapa(x, y), m_mapa(x - 1, y)).eval()) << "notitle with lines lt -1, ";
                if (x != m_mapa.n_rows - 1)
                    gp << gp.file1d(join_vert(m_mapa(x, y), m_mapa(x + 1, y)).eval()) << "notitle with lines lt -1, ";
                if (y != 0)
                    gp << gp.file1d(join_vert(m_mapa(x, y), m_mapa(x, y - 1)).eval()) << "notitle with lines lt -1, ";
                if (y != m_mapa.n_cols - 1)
                    gp << gp.file1d(join_vert(m_mapa(x, y), m_mapa(x, y + 1)).eval()) << "notitle with lines lt -1, ";
            }
        }
    }

    // Título de los centroides para la leyenda
    gp << "NaN title 'Neuronas' with points ps 2 pt 1 lt -1 lw 3" << endl;
}
}
