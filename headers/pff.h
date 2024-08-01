#ifndef PFF_H
#define PFF_H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

#include "cFunctions.h"

class pff
{
private:
    int field;

    bool isPrime(int num);
    // {
    //     if (num <= 1) return false;
    //     if (num <= 3) return true;
    //     if (num % 2 == 0 || num % 3 == 0) return false;
    //     for (int i = 5; i * i <= num; i += 6)
    //         if (num % i == 0 || num % (i + 2) == 0)
    //             return false;
    //     return true;
    // }

public:
    pff(int fieldSize);
    // {
    //     if (isPrime(fieldSize))
    //     {
    //         this->field = fieldSize;
    //     }
    //     else
    //     {
    //         throw std::invalid_argument("Invalid field size: must be a prime number.");
    //     }
    // }

    uint8_t add(uint8_t a, uint8_t b);
    // {
    //     return (a + b) % this->field;
    // }

    uint8_t mutiply(uint8_t a, uint8_t b);
    // {
    //     return (a * b) % this->field;
    // }

    uint8_t mutiplicationInverse(uint8_t a);
    // {
    //     for (int i = 1; i < this->field; i++)
    //         if ((a * i) % this->field == 1)
    //             return i;
    //     return 0;
    // }

    uint8_t additionInverse(uint8_t a);
    // {
    //     return (this->field - a) % this->field;
    // }

    uint8_t division(uint8_t a, uint8_t b);
    // {
    //     if (b == 0)
    //         throw std::invalid_argument("Division by zero is not allowed.");
    //     uint8_t bInverse = this->mutiplicationInverse(b);
    //     return this->multiply(a, bInverse);
    // }

    uint8_t subtraction(uint8_t a, uint8_t b);
    // {
    //     int res = (a - b) % this->field;
    //     if (res < 0)
    //         res += this->field;
    //     return res;
    // }
    std::vector<uint8_t> v2vMulipllication(std::vector<uint8_t> a, std::vector<uint8_t> b);
    std::vector<uint8_t> v2vAddition(std::vector<uint8_t> a, std::vector<uint8_t> b);
    std::vector<uint8_t> v2vSubtraction(std::vector<uint8_t> a, std::vector<uint8_t> b);
    std::vector<uint8_t> s2vMultiplication(std::vector<uint8_t> a, uint8_t b);
    std::vector<uint8_t> s2vDivision(std::vector<uint8_t> a, uint8_t b);
};

#endif