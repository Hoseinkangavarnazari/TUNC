#ifndef CFUNCTIONS_H
#define CFUNCTIONS_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>



enum FieldSize
{
    BinaryA,  // 0
    BinaryB, // 1
    BinaryC
};

std::vector<uint8_t> randomDataGenerator(int dataSize, FieldSize fieldSize);
#endif