#ifndef FF_H
#define FF_H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

#include "cFunctions.h"
// using namespace std;



class ff
{
private:
    FieldSize fieldSize;

public:
    ff(int fieldSize);

    // Each of these will contain the finite field tables
    std::vector<std::vector<uint8_t>> ff256;

    // @TODO: AKIF
    std::vector<std::vector<uint8_t>> ff7;
    std::vector<std::vector<uint8_t>> ff2;

    uint8_t add(uint8_t a, uint8_t b);
    uint8_t mutiply(uint8_t a, uint8_t b);
    uint8_t mutiplicationInverse(uint8_t a);
    uint8_t additionInverse(uint8_t a);
    uint8_t division(uint8_t a, uint8_t b);
    uint8_t subtraction(uint8_t a, uint8_t b);
    std::vector<uint8_t> v2vMulipllication(std::vector<uint8_t> a, std::vector<uint8_t> b);
    std::vector<uint8_t> v2vAddition(std::vector<uint8_t> a, std::vector<uint8_t> b);
    std::vector<uint8_t> v2vSubtraction(std::vector<uint8_t> a, std::vector<uint8_t> b);
    std::vector<uint8_t> s2vMultiplication(std::vector<uint8_t> a, uint8_t b);
    std::vector<uint8_t> s2vDivision(std::vector<uint8_t> a, uint8_t b);
};


#endif
