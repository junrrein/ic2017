#include "radial_por_lotes.cpp"
#include "mlp_salida_lineal.cpp"
#include "../guia1/particionar.cpp"
#include <gnuplot-iostream.h>
#include "../config.hpp"

int main()
{
    arma_rng::set_seed_random();

    mat datos;
    datos.load(config::sourceDir + "/guia2/datos/mervalTuplas.csv");
    const string rutaParticiones = config::sourceDir + "/guia2/datos/particionesMerval/";
    vector<ic::Particion> particiones = ic::cargarParticiones(rutaParticiones, 10);

    ifstream ifs{config::sourceDir + "/guia2/parametrosRbfMerval.txt"};
    ic::ParametrosRBF parametrosRbf;
    if (!(ifs >> parametrosRbf))
        throw runtime_error("Se leyeron mal los parámetros del RBF para el Merval");

    vec erroresPrueba, salidaOrdenada;
    erroresPrueba.set_size(particiones.size());
    salidaOrdenada.set_size(datos.n_rows);
    Gnuplot gp;

    for (unsigned int i = 0; i < particiones.size(); ++i) {
        // Entrenamiento

        mat datosParticion = datos.rows(particiones[i].first);
        mat patrones = datosParticion.head_cols(5);
        mat salidaDeseada = datosParticion.tail_cols(1);

        vector<rowvec> centroides;
        vec sigmas;
        tie(centroides, sigmas) = ic::entrenarRadialPorLotes(patrones,
                                                             parametrosRbf.estructuraRed(0),
                                                             ic::tipoInicializacion::patronesAlAzar);

        mat salidasRadiales(patrones.n_rows, centroides.size());
        for (unsigned int j = 0; j < patrones.n_rows; ++j)
            salidasRadiales.row(j) = ic::salidaRadial(patrones.row(j), centroides, sigmas);

        mat datosParaCapaFinal = join_horiz(salidasRadiales, salidaDeseada);

        vector<mat> pesos;
        double errorEntrenamiento;
        int epocas;
        tie(pesos, errorEntrenamiento, epocas) = ic::entrenarMulticapa(vec{parametrosRbf.estructuraRed(1)},
                                                                       datosParaCapaFinal,
                                                                       parametrosRbf.nEpocas,
                                                                       parametrosRbf.tasaAprendizaje,
                                                                       parametrosRbf.inercia,
                                                                       parametrosRbf.toleranciaError);

        // Prueba

        datosParticion = datos.rows(particiones[i].second);
        patrones = datosParticion.head_cols(5);
        salidaDeseada = datosParticion.tail_cols(1);

        salidasRadiales = mat(patrones.n_rows, centroides.size());
        for (unsigned int j = 0; j < patrones.n_rows; ++j)
            salidasRadiales.row(j) = ic::salidaRadial(patrones.row(j),
                                                      centroides,
                                                      sigmas);

        erroresPrueba(i) = ic::errorRelativoPromedioMulticapa(pesos,
                                                              salidasRadiales,
                                                              salidaDeseada);

        vec salidaRed(patrones.n_rows);
        for (unsigned int n = 0; n < patrones.n_rows; ++n) {
            salidaRed(n) = as_scalar(ic::salidaMulticapa(pesos, salidasRadiales.row(n).t()).back());
        }

        const vec x = linspace(1, salidaRed.n_elem, salidaRed.n_elem);

        gp << "set title 'Predicción de la red en la partición " << i + 1 << " (Prueba)' font ',12'" << endl
           << "set xlabel 'N^{ro} de muestra en la partición'" << endl
           << "set ylabel 'Valor del índice Merval' font ',11'" << endl
           << "set xrange [0:" << salidaRed.n_elem + 1 << "]" << endl
           << "set yrange [0:1200]" << endl
           << "set grid" << endl
           << "set key box opaque" << endl
           << "plot " << gp.file1d(join_horiz(x, salidaDeseada).eval()) << "title 'Salida Deseada' with points pt 6 lt rgb 'red', "
           << gp.file1d(join_horiz(x, salidaDeseada).eval()) << "notitle with impulses lt rgb 'red', "
           << gp.file1d(join_horiz(x, salidaRed).eval()) << "title 'Salida de la Red' with points pt 6 lt rgb 'black', "
           << gp.file1d(join_horiz(x, salidaRed).eval()) << "notitle with impulses lt rgb 'black'" << endl;

        getchar();

        for (unsigned int j = 0; j < salidaRed.n_elem; ++j)
            salidaOrdenada(particiones[i].second.at(j)) = salidaRed(j);
    }

    cout << "Merval RBF, 10 particiones" << endl
         << setprecision(2)
         << "Promedio del error relativo promedio en pruebas: " << mean(erroresPrueba) << " %" << endl
         << "Desvío de lo anterior: " << stddev(erroresPrueba) << " %" << endl;

    vec merval;
    merval.load(config::sourceDir + "/guia2/datos/merval.csv");
    const vec x1 = linspace(0, merval.n_elem - 1, merval.n_elem);
    const vec x2 = linspace(5, 5 + salidaOrdenada.n_elem - 1, salidaOrdenada.n_elem);

    gp << "set title 'Predicción de la red para el Indice Merval (Prueba)' font ',12'" << endl
       << "set xlabel 't (días)' font ',11'" << endl
       << "set ylabel 'valor del índice' font ',11'" << endl
       << "set xrange [0:" << merval.n_elem + 1 << "]" << endl
       << "set yrange [0:1200]" << endl
       << "set grid" << endl
       << "set key box opaque" << endl
       << "plot " << gp.file1d(join_horiz(x1, merval).eval()) << "title 'Salida Deseada' with lines lt rgb 'red', "
       << gp.file1d(join_horiz(x2, salidaOrdenada).eval()) << "title 'Salida de la Red' with lines lw 3 lt rgb 'black'" << endl;

    getchar();

    return 0;
}
