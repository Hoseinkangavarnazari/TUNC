#ifndef RLNC_ENCODER
#define RLNC_ENCODER

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "cFunctions.h"
#include "packet.h"

class rlnc_decoder
{
    int generation_size;
    int symbol_size;
    int systematic_index;
    int decoder_rank;
    FieldSize field_size;

public:
    rlnc_decoder(int g, int s, FieldSize f)
    {
        generation_size = g;
        symbol_size = s;
        field_size = f;
    }

    std::vector<packet> stored_packets;
    std::vector<uint8_t> decode_rlnc(std::vector<uint8_t> coefficients);
    std::vector<uint8_t> decode_complete();
    bool rank();
    bool consumeCodedSymbol(std::vector<uint8_t> coefficient, std::vector<uint8_t> consume_codedSymbol);
    bool consumeSystematicSymbol(std::vector<uint8_t> consume_systematicSymbol);

    bool consumeCodedPacket(packet& p);

    void printAugemntedMatrix();

    // returns true if the decoding is successful
    bool decode();
};

#endif