#include <iostream>
#include <algorithm>
#include "../keygenerator.h"
#include <vector>
#include <random>
#include <ctime>
#include "../ff.h"
#include "../packet.h"

//publickeyset = 
//packet packet();
std::vector<std::vector<uint8_t>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};
std::vector<uint8_t> privatematrix = {3,5,7};
void keygen::publickeygenerator(int _sizeofpublickeys, int _numberofpublickeys){
   publickeyset=matrix;
}

void keygen::privatekeygenerator(int _sizeofprivatekey, int _numberofprivatekeys)
{
    privatekey=privatematrix;
  //_privatekey = dist(0, std::pow(2,fieldSize) , _number_of_mac); 

}


