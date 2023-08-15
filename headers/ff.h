#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
// using namespace std;

class ff
{
private:
    int a = 0;

public:
    std::vector<std::vector<uint8_t>> ff256;

    ff(){
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

        std::cout << "generateTable works!" << std::endl;
    };

    // ~ff();

    uint8_t mutiply(uint8_t a, uint8_t b);
    // uint8_t add(uint8_t a, uint8_t b);
    // uint8_t mutiplicationInverse(uint8_t a);
    // uint8_t additionInverse(uint8_t a);
    // uint8_t division(uint8_t a, uint8_t b);
    // uint8_t subtraction(uint8_t a, uint8_t b);

    // std::vector<uint8_t> v2vMulipllication(std::vector<uint8_t> a, std::vector<uint8_t> b);
    // std::vector<uint8_t> v2vAddition(std::vector<uint8_t> a, std::vector<uint8_t> b);
    // std::vector<uint8_t> v2vSubtraction(std::vector<uint8_t> a, std::vector<uint8_t> b);
    // std::vector<uint8_t> s2vMultiplication(uint8_t a, std::vector<uint8_t> b);


    // std::vector<uint8_t> v2vDivision(std::vector<uint8_t> a, std::vector<uint8_t> b);

};

// std::vector<uint8_t> ff::s2vMultiplication(uint8_t a, std::vector<uint8_t> b){

//     return b;
// };



// uint8_t ff::mutiply(uint8_t a, uint8_t b)
// {

//     return ff256[a][b];
// }

// uint8_t add(uint8_t a, uint8_t b)
// {
//     return a+b%256;
// }