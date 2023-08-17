#include "iostream"
#include "vector"
#include "algorithm"

#include "../cFunctions.h"



std::vector<uint8_t> randomDataGenerator(int dataSize, FieldSize fieldSize)
{

    std::vector<uint8_t> data(dataSize);

    int module = -1;

    switch (fieldSize)
    {
    case BinaryA:
        module = 2;
        break;

    case BinaryB:
        module = 8;
        break;

    case BinaryC:
        module = 256;
        break;

    default:
        std::cerr << "Field size is not defined" << std::endl;
        exit(1);
        break;
    }

    for (size_t i = 0; i < dataSize; i++)
    {
        data[i] = rand() % module;
    }

    return data;
}