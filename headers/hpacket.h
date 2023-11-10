#include <iostream>
#include <vector>
#ifndef HPACKET_H
#define HPACKET_H
#include "packet.h"
#include "ff.h"
#include "keygenerator.h"
#include "mac_calculator.h"
#include "sign_calculator.h"

class hpacket
{
public:
    int packetsize;
    int generationsize;
    std::vector<uint8_t> MACs;
    std::vector<uint8_t> verifier_MACs;
    std::vector<uint8_t> MACs_result;
    std::vector<uint8_t> MACs_multi_verify;
    std::vector<uint8_t> h_codedSymbol;
    int number_of_mac;
    std::vector<std::vector<uint8_t>> publickeyset;
    std::vector<uint8_t> privateKey;
    uint8_t sign;
    uint8_t currentsign;
    std::vector<uint8_t> c_sign;
    std::vector<uint8_t> c_sign_result1;
    uint8_t c_sign_result2;
    uint8_t hosein=0;
    uint8_t hosein_result=0;
    std::vector<uint8_t> mac_result;
    uint8_t sign_result;

   // uint8_t c_sign;

    // int index;
    // std::vector<uint8_t> coeffcients;
    // std::vector<uint8_t> codedSymbol;
    // int symbolsize;
    // std::vector<uint8_t> CRC;
    hpacket(std::vector<uint8_t> _codedSymbol, std::vector<uint8_t> _MAC, std::vector<std::vector<uint8_t>> publickeyset, std::vector<uint8_t> _privateKey, int number_of_mac);
    void macCalculator();
    uint8_t signCalculator();
    bool macVerifier();
    bool signVerifier();

    //};
private:
    std::vector<uint8_t> h_coeffcients;

    std::vector<uint8_t> h_CRC;
    // std::vector<uint8_t> DMAC;
    

    

    // std::vector<uint8_t> &MAC;

   
    
   
    
};

#endif