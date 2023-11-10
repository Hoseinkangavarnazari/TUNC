#include <iostream>
#include <vector>
#ifndef SIGNCALCULATOR_H
#define SIGNCALCULATOR_H
#include "hpacket.h"
#include "ff.h"


class signCalculator
{
   public:
   std::vector<uint8_t> &MAC;
   uint8_t sign;

void signcalculator(std::vector<uint8_t> &_hcodedSymbol,int fieldSize,std::vector<uint8_t> &_privatekey){

  std::vector<uint8_t>private_key = {1,2,3,4,5};
 _privatekey= private_key;
     ff ff(256);
//  First, multiply coefficients[i] with uncodedSymbols[i] and put result in temp
        std::vector<uint8_t> signn = ff.v2vMulipllication(_privatekey, MAC);
        uint8_t sign = signn[1];
        // Second, add temp with codedSymbols and put result in codedSymbols
        sign = ff.division(sign,_privatekey.back());


}
  
};


#endif