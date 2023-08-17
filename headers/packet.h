#ifndef PACKET_H
#define PACKET_H


#include <vector>
#include <iostream>




class packet
{
    public:
    int index;
    std::vector<uint8_t> coeffcients;
    std::vector<uint8_t> codedSymbol;
    std::vector<uint8_t> MAC;
    std::vector<uint8_t> DMAC;
    uint8_t Sign;
    std::vector<uint8_t> CRC;
    packet(std::vector<uint8_t> _coeffcients,std::vector<uint8_t> _codedSymbol, int _index);
};

#endif