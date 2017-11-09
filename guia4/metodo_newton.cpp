#include <armadillo>

using namespace std;
using namespace arma;

double f1(double x)
{
    return -x * sin(sqrt(abs(x)));
}

double f1_x(double x)
{
    return -sin(sqrt(abs(x))) - (x * x * cos(sqrt(abs(x)))) / (2 * pow(abs(x), 3.0 / 2));
}

double f2(double x)
{
    return x + 5 * sin(3 * x) + 8 * cos(5 * x);
}

double f2_x(double x)
{
    return -40 * sin(5 * x) + 15 * cos(3 * x) - 1;
}

double f3(double x, double y)
{
    return -(pow(x * x + y * y, 0.25) * (pow(sin(50 * pow(x * x + y * y, 0.1)), 2)) + 1);
}

double f3_x(double x, double y)
{
    return -(0.5 * x * (pow(sin(50 * pow(y * y + x * x, 0.1)), 2) + 1))
               / pow(y * y + x * x, 0.75)
           - (20.0
              * x
              * cos(50 * pow(y * y + x * x, 0.1))
              * sin(50 * pow(y * y + x * x, 0.1)))
                 / pow(y * y + x * x, 0.65);
}

double f3_y(double x, double y)
{
    return -(0.5
             * y
             * (pow(sin(50 * pow(y * y + x * x, 0.1)), 2) + 1))
               / pow(y * y + x * x, 0.75)
           - (20.0
              * y
              * cos(50 * pow(y * y + x * x, 0.1))
              * sin(50 * pow(y * y + x * x, 0.1)))
                 / pow(y * y + x * x, 0.65);
}

double gradienteDescendente(std::function<double(double)> f_prima,
                            pair<double, double> rangoBusqueda,
                            double alpha)
{
    const double rango = rangoBusqueda.second - rangoBusqueda.first;

    double x = randu(1).eval()(0) * rango + rangoBusqueda.first;

    //    cout << "x = " << x << endl
    //         << "f'(x) =" << f_prima(x) << endl
    //         << "|f'(x)| =" << abs(f_prima(x)) << endl;

    while (abs(f_prima(x)) > 1e-5) {
        x = x - alpha * f_prima(x);

        //        cout << "x = " << x << endl
        //             << "f'(x) =" << f_prima(x) << endl
        //             << "|f'(x)| =" << abs(f_prima(x)) << endl;
    }

    return x;
}

pair<double, double> gradienteDescendente(std::function<double(double, double)> f_x,
                                          std::function<double(double, double)> f_y,
                                          array<pair<double, double>, 2> rangosBusqueda,
                                          double alpha)
{
    const double rangoX = rangosBusqueda.at(0).second - rangosBusqueda.at(0).first;
    const double rangoY = rangosBusqueda.at(1).second - rangosBusqueda.at(1).first;

    double x = randu(1).eval()(0) * rangoX + rangosBusqueda.at(0).first;
    double y = randu(1).eval()(0) * rangoY + rangosBusqueda.at(1).first;

    while (abs(f_x(x, y)) > 1e-5 && abs(f_y(x, y)) > 1e-5) {
        x = x - alpha * f_x(x, y);
        y = y - alpha * f_y(x, y);
    }

    return {x, y};
}
