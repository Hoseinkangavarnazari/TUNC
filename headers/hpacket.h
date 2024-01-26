#include <iostream>
#include <vector>
#ifndef HPACKET_H
#define HPACKET_H
#include "packet.h"
#include "ff.h"
#include "keygenerator.h"
#include "mac_calculator.h"
#include "sign_calculator.h"
#include <cmath>
#include <vector>



class hpacket
{
public:
    uint8_t coefficient;
    int packetsize;
    int generationsize;
    std::vector<std::vector<uint8_t>> MACs;
    std::vector<uint8_t> verifier_MACs;
    std::vector<uint8_t> MACs_result;
    std::vector<std::vector<uint8_t>> h_codedSymbol;
    std::vector<uint8_t> h_combinedSymbol;
    std::vector<std::vector<uint8_t>> h_appendedSymbol;
    int number_of_mac;
    std::vector<uint8_t> MACs_multi_verify;

    std::vector<std::vector<uint8_t>> publickeyset;
    std::vector<uint8_t> privateKey;
    std::vector<uint8_t> sign;
    uint8_t currentsign;
    std::vector<uint8_t> c_sign;
    std::vector<uint8_t> c_sign_result1;
    uint8_t c_sign_result2;
    uint8_t sign_sum = 0;
    uint8_t mac_verifier_sum = 0;
    std::vector<uint8_t> mac_result;
    uint8_t sign_result;
    std::vector<uint8_t> coefficientVector;
    uint8_t combination_counter;

    //int verification_counter=0;
    std::vector<std::vector<uint8_t>> verified_symbols;
    bool verification_resultsFlag = true;
    uint8_t generator = 2;
    uint8_t signMultiply =1;
    uint8_t multiple_sum;
    std::vector<uint8_t> mult_result;
    std::vector<uint8_t> div_result;
    std::vector<uint8_t> mult_inv_result;
    std::vector<uint8_t> verifiedDataPacket;
    std::vector<std::vector<uint8_t>> received_packets_list;
    uint8_t vectorSize;
    //std::vector<uint8_t> zeroVector;
    uint8_t number_of_leaves;
    uint8_t number_of_layers;  //Except the bottom layer
    std::vector<uint8_t>add_inv_table;
    std::vector<std::vector<std::vector<uint8_t>>> generatedTree;


//TreeGenerator(std::vector<std::vector<uint8_t>> received_packets_list);



    // uint8_t c_sign;

    // int index;
    // std::vector<uint8_t> coeffcients;
    // std::vector<uint8_t> codedSymbol;
    // int symbolsize;
    // std::vector<uint8_t> CRC;
    hpacket(std::vector<std::vector<uint8_t>> _codedSymbol, std::vector<std::vector<uint8_t>> _MAC, std::vector<std::vector<uint8_t>> _publickeyset, std::vector<uint8_t> _privateKey, int number_of_mac,std::vector<uint8_t> _coefficientvector);
    void macCalculator();
    void signCalculator();
    //bool Verifier();
    bool macVerifier(std::vector<uint8_t> verifiedDataPacket);
    bool signVerifier(std::vector<uint8_t> verifiedDataPacket);
    void packetCombiner();
    std::vector<std::vector<uint8_t>> packetAppender(std::vector<std::vector<uint8_t>> _h_appendedSymbol);
    uint8_t powerCalculator(uint8_t k ,uint8_t n);
    void multiplyCheck();
    std::vector<std::vector<std::vector<uint8_t>>>  treeGenerator(std::vector<std::vector<uint8_t>> received_packets_list, int _numberOfLayers,int nodes,int leaves);
    std::vector<std::vector<uint8_t>> pollutionGeneration(std::vector<std::vector<uint8_t>> received_packets_list,std::vector<int> pollutedPacketIndex);
    int treeVerifier(std::vector<std::vector<std::vector<uint8_t>>> received_packets_tree,int _layer,int _leaves);
    int arTreeVerifier(std::vector<std::vector<std::vector<uint8_t>>> received_packets_tree,std::vector<int> ARvector,int _layer);
    int simpleVerifier(std::vector<std::vector<uint8_t>> received_packets_list);

   
   
    //};
private:
    std::vector<uint8_t> h_coeffcients;

    std::vector<uint8_t> h_CRC;
    // std::vector<uint8_t> DMAC;

    // std::vector<uint8_t> &MAC;
};

#endif