#ifndef RLNC_DECODER
#define RLNC_DECODER

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "cFunctions.h"

class rlnc_encoder
{
    int generation_size;
    int symbol_size;
    int systematic_index;
    FieldSize field_size;
    rlnc_encoder(int g, int s, FieldSize f)
    {
        generation_size = g;
        symbol_size = s;
        field_size = f;
    }
    bool setSymbol();
    std::vector<uint8_t> encode_systematic();
    std::vector<uint8_t> encode_rlnc(std::vector<uint8_t> coefficients);
    std::vector<uint8_t> returnSymbol();

    void randomCoeffGenerator(std::vector<uint8_t> &vec);
};

#endif