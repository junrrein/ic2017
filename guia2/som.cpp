#include <armadillo>

using namespace arma;
using namespace std;

class SOM {
public:
    // Constructores
    SOM(const mat& patrones, pair<int, int> dimensiones);

    // Interfaz
    void entrenar(int nEpocas,
                  double velocidadInicial,
                  double velocidadFinal,
                  int vecindadInicial,
                  int vecindadFinal);

    // Acceso a miembros
    const field<rowvec>& mapa() const { return m_mapa; };
    const mat& patrones() const { return m_patrones; };

private:
    field<rowvec> m_mapa;
    const mat m_patrones;
};

SOM::SOM(const mat& patrones, pair<int, int> dimensiones)
    : m_patrones{patrones}
{
    m_mapa = field<rowvec>(dimensiones.first, dimensiones.second);

    // Inicializar mapa
    for (unsigned int i = 0; i < m_mapa.n_rows; ++i) {
        for (unsigned int j = 0; j < m_mapa.n_cols; ++j) {
            m_mapa(i, j) = randu(patrones.n_cols) - 0.5;
        }
    }
}

void SOM::entrenar(int nEpocas,
                   double velocidadInicial,
                   double velocidadFinal,
                   int vecindadInicial,
                   int vecindadFinal)
{
    for (int i = 0; i < nEpocas; ++i) {
        for (unsigned int n = 0; n < m_patrones.n_rows; ++n) {

            // Buscamos la neurona ganadora
            pair<int, int> coordGanadora;
            double distanciaGanadora = numeric_limits<double>::max();

            for (unsigned int j = 0; j < m_mapa.n_rows; ++j) {
                for (unsigned int k = 0; k < m_mapa.n_rows; ++k) {
                    const double distancia = norm(m_patrones.row(n) - m_mapa(j, k));

                    if (distancia < distanciaGanadora) {
                        distanciaGanadora = distancia;
                        coordGanadora = {j, k};
                    }
                }
            }

            // Adaptación de pesos
            int vecindad = 1;
            double velocidad = 0.2;

            const int xInicial = (coordGanadora.first - vecindad < 0) ? 0 : (coordGanadora.first - vecindad);
            const int xFinal = (coordGanadora.first + vecindad > m_mapa.n_cols) ? m_mapa.n_cols : (coordGanadora.first + vecindad);
            const int yInicial = (coordGanadora.second - vecindad < 0) ? 0 : (coordGanadora.second - vecindad);
            const int yFinal = (coordGanadora.second + vecindad > m_mapa.n_rows) ? m_mapa.n_rows : (coordGanadora.second + vecindad);

            for (int x = xInicial; x <= xFinal; ++x) {
                for (int y = yInicial; y <= yFinal; ++y) {
                    m_mapa(x, y) += velocidad * distanciaGanadora;
                }
            }
        }
    }
}
