#ifndef PACKET_H
#define PACKET_H

#include <vector>
#include <iostream>

class packet
{
public:
    int index;
    int packetsize;
    int generationsize;
    packet()
    {
        this->index = 1;
    };
    
    std::vector<uint8_t> coeffcients;
    std::vector<uint8_t> codedSymbol;

    std::vector<uint8_t> CRC;
    packet(std::vector<uint8_t> _coeffcients, std::vector<uint8_t> _codedSymbol, int _index, int _packetsize, int _generationsize);
};

#endif