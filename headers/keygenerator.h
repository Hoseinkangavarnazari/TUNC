#ifndef KEYGENERATOR_H
#define KEYGENERATOR_H
#include <iostream>
#include <vector>
#include <string>
#include "packet.h"




class keygen
{
    private:
    std::vector<uint8_t> privatekey ; 
    int number_of_mac;

    public:

    std::vector<std::vector<uint8_t>> publickeyset;

    void publickeygenerator(int _sizeofpublickeys, int _numberofpublickeys);

    void privatekeygenerator(int _sizeofprivatekey, int _numberofprivatekeys);


    

    // Generate a random key.
//    return dist(gen);

};





#endif