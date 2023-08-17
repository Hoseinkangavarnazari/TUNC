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

int main()
{

    ff ff(256);

    int symbolSize = 4;
    int generationSize = 5;

    FieldSize fieldSize = BinaryC;
    int dataSizeInBytes = generationSize * symbolSize;
    std::vector<uint8_t> sampleData = randomDataGenerator(dataSizeInBytes, fieldSize);

    rlnc_encoder encoder(generationSize, symbolSize, fieldSize);
    rlnc_decoder decoder(generationSize, symbolSize, fieldSize);


    bool res = encoder.setSymbol(sampleData);

    // while

    std::vector<packet> allPackets;

    int packetCount = 0;
    while (packetCount != generationSize)
    {   
        // create a coded packet
        // First, generate a random coefficient
        std::vector<uint8_t> tempCoeff;
        encoder.randomCoeffGenerator(tempCoeff);
        // Second, encode using the coeff
        std::vector<uint8_t> codedSybmols = encoder.encode_rlnc(tempCoeff);
        // Third, create a packet object
        packet p(tempCoeff, codedSybmols,packetCount);


        // give the packet to the decoder 
        allPackets.push_back(p);
        decoder.consumeCodedPacket(p);
        packetCount++;
    }

    decoder.decode();


    
    std::cout << "Here " << std::endl;
    return 0;
    std::vector<packet> packets;

    return 0;
}
