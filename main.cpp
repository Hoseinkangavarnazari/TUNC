#include <iostream>
#include "headers/ff.h"
#include <vector>
#include <stdlib.h>

int main()
{
    std::cout << "Hello World!\n";

    ff ff(256);

    std::vector<uint8_t> a = { 11, 12, 3, 43 };
    std::vector<uint8_t> b = { 51, 61, 23, 8 };
    std::vector<uint8_t> c = ff.v2vSubtraction(a, b);
    return 0;
}
