
#include "../packet.h"

packet::packet(std::vector<uint8_t> _coeffcients, std::vector<uint8_t> _codedSymbol, int _index, int _packetsize, int _generationsize)

{
    coeffcients = _coeffcients;
    codedSymbol = _codedSymbol;
    index = _index;
    generationsize = _generationsize;
    packetsize= _packetsize;
}
