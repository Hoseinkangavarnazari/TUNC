#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
// using namespace std;

enum FieldSize
{
    BinaryA = 2,  // 0
    BinaryB = 8, // 1
    BinaryC = 256, // 2
};

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
    std::vector<uint8_t> s2vMultiplication(uint8_t a, std::vector<uint8_t> b);
};
