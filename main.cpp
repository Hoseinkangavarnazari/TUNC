#include <iostream>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <random>

#include "headers/rlnc_encoder.h"
#include "headers/ff.h"
#include "headers/cFunctions.h"



int main()
{
    std::vector<uint8_t> vec(5);
    std::cout << "Hello World!\n";
    FieldSize fieldSize = BinaryC;
    std::vector<uint8_t> test =  randomDataGenerator(1000 , fieldSize);

    // ff ff(256);

    // std::vector<uint8_t> a = {2, 3, 10, 5};
    // std::vector<uint8_t> b = {22, 3, 10, 5};
    // std::vector<uint8_t> c = {2, 3, 15, 3};
    // std::vector<uint8_t> d = {0, 3, 10, 5};
    // std::vector<std::vector<uint8_t>> matrix;
    // matrix.push_back(a);
    // matrix.push_back(b);
    // matrix.push_back(c);
    // matrix.push_back(d);

    // std::vector<uint8_t> c = ff.s2vDivision(a, b);

    return 0;
}
