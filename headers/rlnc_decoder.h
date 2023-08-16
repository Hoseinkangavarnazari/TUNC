#include <vector>
#include <iostream>
#include <fstream>
#include <string>


enum FieldSize
{
    BinaryA = 2,
    BinaryB = 8,
    BinaryC = 256,
};


class rlnc_decoder
{
    int generation_size;
    int symbol_size;
    int systematic_index;
    FieldSize field_size;
    rlnc_decoder(int g, int s, FieldSize f)
    {
        generation_size = g;
        symbol_size = s;
        field_size = f;
    }
    bool setSymbol();

    std::vector<uint8_t> decode_rlnc(std::vector<uint8_t> coefficients);
    std::vector<uint8_t> decode_complete();

    int decoder_rank(); 
    bool consume_codedSymbol(std::vector<uint8_t> coefficient, std::vector<uint8_t> consume_codedSymbol);
    bool consume_systematicSymbol( std::vector<uint8_t> consume_systematicSymbol);
};