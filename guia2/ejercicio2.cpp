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
        throw runtime_error("Se leyeron mal los parámetros del RBF para el Iris");

    // Estructura de datos para guardar los pesos de todas las redes
    // (una para cada partición)
    //    using Centroides = vector<rowvec>;
    //    vector<Centroides> centroidesRedesRbf;
    //    vector<vec> sigmasRedesRbf;
    //    using Pesos = vector<mat>;
    //    vector<Pesos> pesosCapasFinalesRbf;
    //    centroidesRedesRbf.resize(particiones.size());
    //    sigmasRedesRbf.resize(particiones.size());
    //    pesosCapasFinalesRbf.resize(particiones.size());

    //    for (unsigned int i = 0; i < particiones.size(); ++i) {
    //        const mat datosParticion = datos.rows(particiones[i].first);
    //        const mat patrones = datosParticion.head_cols(4);
    //        const mat salidaDeseada = datosParticion.tail_cols(3);

    //        tie(centroidesRedesRbf[i], sigmasRedesRbf[i]) = ic::entrenarRadialPorLotes(patrones,
    //                                                                                   parametrosRbf.estructuraRed(0),
    //                                                                                   ic::tipoInicializacion::patronesAlAzar);

    //        mat salidasRadiales(patrones.n_rows, centroidesRedesRbf[i].size());
    //        for (unsigned int j = 0; j < patrones.n_rows; ++j)
    //            salidasRadiales.row(j) = ic::salidaRadial(patrones.row(j), centroidesRedesRbf[i], sigmasRedesRbf[i]);

    //        const mat datosParaCapaFinal = join_horiz(salidasRadiales, salidaDeseada);

    //        tie(pesosCapasFinalesRbf[i], ignore, ignore) = ic::entrenarMulticapa(vec{parametrosRbf.estructuraRed(1)},
    //                                                                             datosParaCapaFinal,
    //                                                                             parametrosRbf.nEpocas,
    //                                                                             parametrosRbf.tasaAprendizaje,
    //                                                                             parametrosRbf.inercia,
    //                                                                             parametrosRbf.toleranciaError);
    //    }

    //    // Cálculo del error del RBF
    //    vec erroresRbf;
    //    erroresRbf.set_size(particiones.size());

    //    for (unsigned int i = 0; i < particiones.size(); ++i) {
    //        const mat datosParticion = datos.rows(particiones[i].second);
    //        const mat patrones = datosParticion.head_cols(4);
    //        const mat salidaDeseada = datosParticion.tail_cols(3);

    //        mat salidasRadiales(patrones.n_rows, centroidesRedesRbf[i].size());
    //        for (unsigned int j = 0; j < patrones.n_rows; ++j)
    //            salidasRadiales.row(j) = ic::salidaRadial(patrones.row(j),
    //                                                      centroidesRedesRbf[i],
    //                                                      sigmasRedesRbf[i]);

    //        const mat datosParaCapaFinal = join_horiz(salidasRadiales, salidaDeseada);

    //        erroresRbf(i) = ic::errorCuadraticoMulticapa(pesosCapasFinalesRbf[i],
    //                                                     salidasRadiales,
    //                                                     salidaDeseada);
    //    }

    //    cout << "Merval RBF" << endl
    //         << "Error cuadrático promedio en pruebas: " << mean(erroresRbf) << endl
    //         << "Desvío de lo anterior: " << stddev(erroresRbf) << endl;

    mat datosParticion = datos.rows(particiones[1].first);
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
    int epocas;
    vec errores;
    tie(pesos, errores, epocas) = ic::entrenarMulticapa(vec{parametrosRbf.estructuraRed(1)},
                                                        datosParaCapaFinal,
                                                        parametrosRbf.nEpocas,
                                                        parametrosRbf.tasaAprendizaje,
                                                        parametrosRbf.inercia,
                                                        parametrosRbf.toleranciaError * patrones.n_rows);

    // Cálculo del error del RBF
    datosParticion = datos.rows(particiones[1].second);
    patrones = datosParticion.head_cols(5);
    salidaDeseada = datosParticion.tail_cols(1);

    salidasRadiales = mat(patrones.n_rows, centroides.size());
    for (unsigned int j = 0; j < patrones.n_rows; ++j)
        salidasRadiales.row(j) = ic::salidaRadial(patrones.row(j),
                                                  centroides,
                                                  sigmas);

    const double error = ic::errorCuadraticoMulticapa(pesos,
                                                      salidasRadiales,
                                                      salidaDeseada);

    vec salidaRed(patrones.n_rows);
    for (unsigned int n = 0; n < patrones.n_rows; ++n) {
        salidaRed(n) = as_scalar(ic::salidaMulticapa(pesos, salidasRadiales.row(n).t()).back());
    }

    cout << "Error cuadrático promedio prueba: " << error / 47 << endl
         << "Error cuadrático promedio entrenamiento: " << errores(errores.n_elem - 1) / 426 << endl
         << "Epocas: " << epocas << endl;

    Gnuplot gp;
    gp << "plot " << gp.file1d(salidaDeseada) << "title 'Salida Deseada' with lines, "
       << gp.file1d(salidaRed) << "title 'Salida de la Red' with lines lw 2" << endl;

    return 0;
}
