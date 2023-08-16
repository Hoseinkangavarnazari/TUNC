
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

enum FieldSize
{
    Binary,  // 0
    Binary8, // 1
    Binary256
};

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
};