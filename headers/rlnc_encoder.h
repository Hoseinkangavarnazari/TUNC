#ifndef RLNC_DECODER
#define RLNC_DECODER

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "cFunctions.h"
#include "ff.h"


class rlnc_encoder
{
private:
    int generation_size;
    int symbol_size;
    int systematic_index;

    std::vector<std::vector<uint8_t>> uncodedSymbols;

public:

    FieldSize field_size;
    
    rlnc_encoder(int g, int s, FieldSize f);
    // {
    //     generation_size = g;
    //     symbol_size = s;
    //     field_size = f;
    //     uncodedSymbols.resize(g);

    //     for (int i = 0; i < g; i++)
    //     {
    //         uncodedSymbols[i].resize(s);
    //         uncodedSymbols[i].assign(s, 0);
    //     }
    // }
    bool setSymbol(std::vector<uint8_t> data);
    std::vector<uint8_t> encode_systematic();
    std::vector<uint8_t> encode_rlnc(std::vector<uint8_t> coefficients);
    std::vector<uint8_t> returnSymbol();

    void randomCoeffGenerator(std::vector<uint8_t> &vec);
};




#endif