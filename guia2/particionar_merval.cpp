#include "../guia1/particionar.cpp"
#include "../config.hpp"
#include <boost/filesystem.hpp>

int main()
{
    arma_rng::set_seed_random();

    vec datos;
    datos.load(config::sourceDir + "/guia2/datos/merval.csv");

    mat tuplas(datos.n_elem - 5, 6);

    for (unsigned int i = 0; i < datos.n_rows - 5; ++i) {
        tuplas.row(i) = datos.rows(span(i, i + 5)).t();
    }

    // FIXME: ¿Hay que randomizar el orden de las tuplas?

    tuplas.save(config::sourceDir + "/guia2/datos/mervalTuplas.csv", csv_ascii);

    vector<ic::Particion> particiones = ic::leaveKOut(tuplas, 47);

    boost::filesystem::create_directory(config::sourceDir + "/guia2/datos/particionesMerval/");
    ic::guardarParticiones(particiones, config::sourceDir + "/guia2/datos/particionesMerval/");

    return 0;
}
