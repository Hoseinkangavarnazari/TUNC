
#include "../packet.h"

packet::packet(std::vector<uint8_t> _coeffcients, std::vector<uint8_t> _codedSymbol, int _index)
{
    coeffcients = _coeffcients;
    codedSymbol = _codedSymbol;
    index = _index;
}
