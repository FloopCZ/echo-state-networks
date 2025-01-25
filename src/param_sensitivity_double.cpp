// Parameter sensitivity evaluation for integer parameters.

#include "param_sensitivity.hpp"

int main(int argc, char* argv[])
{
    return param_sensitivity<double>(argc, argv);
}
