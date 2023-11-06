#include <iostream>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <random>

#include "headers/rlnc_encoder.h"
#include "headers/ff.h"
#include "headers/cFunctions.h"
#include "headers/packet.h"
#include "headers/rlnc_decoder.h"
#include "headers/hpacket.h"

//hpacket macCalculator();

int main()
{
    std::vector<std::vector<uint8_t>> publickeyset = {{1,2,3},{2,3,4},{3,4,5},{4,5,6},{5,6,7}};
    std::vector<std::vector<uint8_t>> hcodedSymbol = {{2,4},{1,3},{6,7}};          
    //hpacket packettemp ;
      ff ff(256);
    
    int symbolSize = 2;
    int generationSize = 6;
    //std::vector<std::vector<uint8_t>> test_mac;
    hpacket hp( hcodedSymbol, publickeyset, publickeyset.size());
    //test_mac = macCalculator(hcodedSymbol, publickeyset, 5);
    FieldSize fieldSize = BinaryC;
    int dataSizeInBytes = generationSize * symbolSize;
    std::vector<uint8_t> sampleData = randomDataGenerator(dataSizeInBytes, fieldSize);
    std::vector<std::vector<uint8_t>>test_mac = hp.macCalculator(hcodedSymbol,publickeyset,publickeyset.size());
    rlnc_encoder encoder(generationSize, symbolSize, fieldSize);
    rlnc_decoder decoder(generationSize, symbolSize, fieldSize);


    bool res = encoder.setSymbol(sampleData);

    // while

    std::vector<packet> allPackets;

    int packetCount = 0;
    //    while (packetCount != generationSize)
    //  {   
// create a coded packet
// First, generate a random coefficient
    //    std::vector<uint8_t> tempCoeff;
      //  encoder.randomCoeffGenerator(tempCoeff);
// Second, encode using the coeff
       // std::vector<uint8_t> codedSybmols = encoder.encode_rlnc(tempCoeff);
// Third, create a packet object
        //packet p(tempCoeff, codedSybmols,packetCount);


        // give the packet to the decoder 
        //allPackets.push_back(p);
        //decoder.consumeCodedPacket(p);
        //packetCount++;
    //}

    //decoder.decode();


    
    std::cout << "Here " << std::endl;
    return 0;
    std::vector<packet> packets;

    return 0;
}
