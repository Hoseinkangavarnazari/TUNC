#include "../ff.h"
#include <algorithm>
#include <iostream>

uint8_t ff::mutiply(uint8_t a, uint8_t b)
{
    return ff256[a][b];
}

uint8_t ff::add(uint8_t a, uint8_t b)
{
    

    int module = -1;

    switch (this->fieldSize)
    {
    case BinaryA:
        module = 2;
        break;

    case BinaryB:
        module = 8;
        break;

    case BinaryC:
        module = 256;
        break;

    default:
        return EXIT_FAILURE;
        break;
    }

    // return a + b % module;
    return a ^ b;
}

uint8_t ff::mutiplicationInverse(uint8_t a)
{
    // std::cout << "mutiplicationInverse Endpoint hit" << std::endl;

    int index = -1;

    if (this->fieldSize == BinaryC)
    {
        auto it = std::find(ff256[a].begin(), ff256[a].end(), 1);

        // if there is a one inside that row at all 
        if (it != ff256[a].end())
        {
            index = std::distance(ff256[a].begin(), it);
        }
    }

    return static_cast<uint8_t>(index);
};
uint8_t ff::additionInverse(uint8_t a)
{
    return a;
};
//  a is divided by b
uint8_t ff::division(uint8_t a, uint8_t b)
{
    uint8_t bInverse= this->mutiplicationInverse(b);
    uint8_t result = this->mutiply(a,bInverse);
    return result;
};
uint8_t ff::subtraction(uint8_t a, uint8_t b)
{
    return a;
};
std::vector<uint8_t> ff::v2vMulipllication(std::vector<uint8_t> a, std::vector<uint8_t> b)
{
    return a;
};
std::vector<uint8_t> ff::v2vAddition(std::vector<uint8_t> a, std::vector<uint8_t> b)
{
    return a;
};
std::vector<uint8_t> ff::v2vSubtraction(std::vector<uint8_t> a, std::vector<uint8_t> b)
{
    return a;
};
std::vector<uint8_t> ff::s2vMultiplication(uint8_t a, std::vector<uint8_t> b)
{
    return b;
};

ff::ff(int fieldSize)
{

    // fill out the tables for fields from files
    std::ifstream inputFile("result.txt");

    if (!inputFile.is_open())
    {
        std::cerr << "Could not open the file." << std::endl;
    }

    std::string line;
    while (std::getline(inputFile, line))
    {

        std::vector<uint8_t> ff256_row;

        while (line.end() - line.begin() > 1)
        {
            // std::cout << line << std::endl;
            size_t pos = line.find("-");

            std::string temp = line.substr(0, pos);
            ff256_row.push_back(std::stoi(temp));

            line.erase(0, pos + 1);
        }
        ff256.push_back(ff256_row);
    }

    // ...........................................

    if (fieldSize == 2)
    {
        this->fieldSize = BinaryA;
    }
    else if (fieldSize == 8)
    {
        this->fieldSize = BinaryB;
    }
    else if (fieldSize == 256)
    {
        this->fieldSize = BinaryC;
    }
    else
    {
        std::cerr << "FATAL system error: Invalid field size" << std::endl;
        exit(EXIT_FAILURE);
    }
};