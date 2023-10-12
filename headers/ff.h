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
    std::vector<std::vector<uint8_t>> ff256_addition;
    std::vector<std::vector<uint8_t>> ff256_multiplication;
    // @TODO: AKIF
    std::vector<std::vector<uint8_t>> ff7_addition;
    std::vector<std::vector<uint8_t>> ff7_multiplication;
    std::vector<std::vector<uint8_t>> ff2_addition;
    std::vector<std::vector<uint8_t>> ff2_multiplication;


// Function to perform addition modulo the field size
int addInField(uint8_t a, uint8_t b, int fieldSize) {
    return (a + b) % fieldSize;
}

int main() {
    int fieldSize = 16;

    // Create a 2D vector to represent the addition table and initialize its dimensions
    std::vector<std::vector<int>> ff_addition(fieldSize, std::vector<int>(fieldSize));

    // Fill in the addition table
    for (uint8_t i = 0; i < fieldSize; i++) {
        for (uint8_t j = 0; j < fieldSize; j++) {
            ff_addition[i][j] = addInField(i, j, fieldSize);
        }
    }

    // Display the addition table
    for (uint8_t i = 0; i < fieldSize; i++) {
        for (uint8_t j = 0; j < fieldSize; j++) {
            std::cout << ff_addition[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}





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
