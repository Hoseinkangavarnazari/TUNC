#include <iostream>
#include "headers/ff.h"
#include <vector>
#include <stdlib.h>

int main()
{
    std::cout << "Hello World!\n";

    ff ff(256);

    std::vector<uint8_t> a = {110, 120, 30, 43};

    uint8_t b = 255;

    std::vector<uint8_t> c = ff.s2vDivision(a, b);

    return 0;
}
