#include "../rlnc_encoder.h"
#include <iostream>
#include <algorithm>
#include <cassert>
#include "../ff.h"

void rlnc_encoder::randomCoeffGenerator(std::vector<uint8_t> &vec)
{
    vec.resize(this->generation_size);
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

        zeros = std::all_of(vec.begin(),vec.end(), [](int i)
                            { return i == 0; });
    }
};

std::vector<uint8_t> rlnc_encoder::encode_rlnc(std::vector<uint8_t> coefficients)
{
    // you need to multiply each coefficient with uncoded symbol and add them together
    // finally you need to return the result

    ff ff(256);
    std::vector<uint8_t> codedSymbols(this->symbol_size, 0);

    std::vector<uint8_t> temp(this->symbol_size, 0);
    for (size_t i = 0; i < coefficients.size(); i++)
    {

        //  First, multiply coefficients[i] with uncodedSymbols[i] and put result in temp
        temp = ff.s2vMultiplication(this->uncodedSymbols[i], coefficients[i]);

        // Second, add temp with codedSymbols and put result in codedSymbols
        codedSymbols = ff.v2vAddition(codedSymbols, temp);
    }
    return codedSymbols;
}

bool rlnc_encoder::setSymbol(std::vector<uint8_t> data)
{

    bool res = this->generation_size * this->symbol_size == data.size() ? true : false;
    assert(this->generation_size * this->symbol_size == data.size());

    for (size_t i = 0; i < this->generation_size; i++)
    {
        for (size_t j = 0; j < this->symbol_size; j++)
        {
            this->uncodedSymbols[i][j] = data[i * this->symbol_size + j];
        }
    }

    return res;
}

rlnc_encoder::rlnc_encoder(int g, int s, FieldSize f)
{
    generation_size = g;
    symbol_size = s;
    field_size = f;
    uncodedSymbols.resize(g);

    for (int i = 0; i < g; i++)
    {
        uncodedSymbols[i].resize(s);
        uncodedSymbols[i].assign(s, 0);
    }
}