#include "../rlnc_encoder.h"


#include <iostream>
#include <algorithm>

void rlnc_encoder::randomCoeffGenerator(std::vector<uint8_t> &vec)
{

    std::fill(vec.begin(), vec.end(), 0);
    bool zeros = std::all_of(vec.begin(), vec.end(), [](int i)
                             { return i == 0; });

    int module = -1;

    switch (this->field_size)
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
        return;
        break;
    }

    while (zeros)
    {
        for (size_t i = 0; i < vec.size(); i++)
        {
            vec[i] = rand() % module;
        }

        zeros = std::all_of(vec.begin(), vec.end(), [](int i)
                            { return i == 0; });
    }
};