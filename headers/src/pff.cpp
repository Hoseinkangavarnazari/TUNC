#include "../pff.h"
#include <algorithm>
#include <iostream>

 

bool pff::isPrime(int num)
    {
        if (num <= 1) return false;
        if (num <= 3) return true;
        if (num % 2 == 0 || num % 3 == 0) return false;
        for (int i = 5; i * i <= num; i += 6)
            if (num % i == 0 || num % (i + 2) == 0)
                return false;
        return true;
    }

pff::pff(int fieldSize)
    {
        if (isPrime(fieldSize))
        {
            this->field = fieldSize;
        }
        else
        {
            throw std::invalid_argument("Invalid field size: must be a prime number.");
        }
    }


 uint8_t pff::add(uint8_t a, uint8_t b)
    {
        return (a + b) % this->field;
    }

    uint8_t pff::mutiply(uint8_t a, uint8_t b)
    {
        return (a * b) % this->field;
    }

    uint8_t pff::mutiplicationInverse(uint8_t a)
    {
        for (int i = 1; i < this->field; i++)
            if ((a * i) % this->field == 1)
                return i;
        return 0;
    }

    uint8_t pff::additionInverse(uint8_t a)
    {
        return (this->field - a) % this->field;
    }

    uint8_t pff::division(uint8_t a, uint8_t b)
    {
        //if (b == 0){
//            throw std::invalid_argument("Division by zero is not allowed.");
        uint8_t bInverse = this->mutiplicationInverse(b);
        //};
        return this->mutiply(a, bInverse);
    }

    uint8_t pff::subtraction(uint8_t a, uint8_t b)
    {
        int res = (a - b) % this->field;
        if (res < 0)
            res += this->field;
        return res;
    };

    std::vector<uint8_t> pff::v2vMulipllication(std::vector<uint8_t> a, std::vector<uint8_t> b){
        std::vector<uint8_t> answer;

    if (a.size() != b.size())
    {
        std::cerr << "FATAL system error: Invalid vector sizes" << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        for (size_t i = 0; i < a.size(); i++)
        {
            answer.push_back(this->mutiply(a[i], b[i]));
        }
    }
    return answer;

    };
    std::vector<uint8_t> pff::v2vAddition(std::vector<uint8_t> a, std::vector<uint8_t> b){
        std::vector<uint8_t> answer;

    if (a.size() != b.size())
    {
        std::cerr << "FATAL system error: Invalid vector sizes" << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        for (size_t i = 0; i < a.size(); i++)
        {
            answer.push_back(add(a[i],b[i]));
        }
    }
    return answer;

    };
    std::vector<uint8_t> pff::v2vSubtraction(std::vector<uint8_t> a, std::vector<uint8_t> b){
         std::vector<uint8_t> answer;

    if (a.size() != b.size())
    {
        std::cerr << "FATAL system error: Invalid vector sizes" << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        for (size_t i = 0; i < a.size(); i++)
        {
            answer.push_back(subtraction(a[i], b[i]));
        }
    }
    return answer;

    };
    std::vector<uint8_t> pff::s2vMultiplication(std::vector<uint8_t> a, uint8_t b){
         std::vector<uint8_t> answer;

    for (int i = 0; i < a.size(); i++)
    {
        answer.push_back(this->mutiply(a[i],b));
    }
    return answer;

    };
    std::vector<uint8_t> pff::s2vDivision(std::vector<uint8_t> a, uint8_t b){
            std::vector<uint8_t> answer;

    for (int i = 0; i < a.size(); i++)
    {
        answer.push_back(this->division(a[i],b));
    }
    return answer;   

    };