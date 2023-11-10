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
#include <chrono>

std::vector<uint8_t> generateRandomVector(int size)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dis(0, 255);

  std::vector<uint8_t> randomVector(size);

  for (int i = 0; i < size; ++i)
  {
    randomVector[i] = dis(gen);
  }

  return randomVector;
}
// hpacket macCalculator();

int main()
{
  // std::vector<std::vector<uint8_t>> publickeyset = {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}, {5, 6, 7}};
  // std::vector<std::vector<uint8_t>> hcodedSymbol = {{2, 4}, {1, 3}, {6, 7}};
  // // hpacket packettemp ;
  // ff ff(256);

  // int symbolSize = 3;
  // int generationSize = 4;
  // // std::vector<std::vector<uint8_t>> test_mac;
  // hpacket hp(hcodedSymbol, publickeyset, publickeyset.size());
  // // test_mac = macCalculator(hcodedSymbol, publickeyset, 5);
  // FieldSize fieldSize = BinaryC;

  // int dataSizeInBytes = generationSize * symbolSize;
  // std::vector<uint8_t> sampleData = randomDataGenerator(dataSizeInBytes, fieldSize);
  // std::vector<std::vector<uint8_t>> test_mac = hp.macCalculator(hcodedSymbol, publickeyset, publickeyset.size());
  // rlnc_encoder encoder(generationSize, symbolSize, fieldSize);
  // rlnc_decoder decoder(generationSize, symbolSize, fieldSize);

  // bool res = encoder.setSymbol(sampleData);

  // // while

  // std::vector<packet> allPackets;

  // int packetCount = 0;
  // while (packetCount != generationSize)
  // {
  //   // create a coded packet
  //   // First, generate a random coefficient
  //   std::vector<uint8_t> tempCoeff;
  //   encoder.randomCoeffGenerator(tempCoeff);
  //   // Second, encode using the coeff
  //   std::vector<uint8_t> codedSybmols = encoder.encode_rlnc(tempCoeff);
  //   // Third, create a packet object
  //   // packet p(tempCoeff, codedSybmols,packetCount);

  //   // give the packet to the decoder
  //   // allPackets.push_back(p);
  //   // decoder.consumeCodedPacket(p);
  //   // packetCount++;
  // }

  // decoder.decode();

  std::vector<std::vector<uint8_t>> key1 = {{2, 3, 1, 4, 2}};
  std::vector<uint8_t> private_key1 = {2, 4};
  // std::vector<uint8_t> private_key2 = {1, 7, 123, 8};
  // std::vector<uint8_t> private_key3 = {55, 58, 71};
  std::vector<uint8_t> private_key = {5, 4, 3, 2, 1};

  std::vector<uint8_t> MACs;
  uint8_t currentsign;
  // uint8_t c_sign;
  uint8_t sign;

  // std::vector<uint8_t> cs2 = {7, 4, 2};
  // std::vector<std::vector<uint8_t>> key2 = {{2, 3, 1, 4},{2, 3, 1, 3},{1,2,3,4}};
  // hpacket p2(cs2, MACs, key2, private_key2, 3);

  // std::vector<uint8_t> cs3 = {8, 4};
  // std::vector<std::vector<uint8_t>> key3 = {{2, 3, 1},{2, 1, 7}};
  // hpacket p3(cs3, MACs, key3, private_key3, 2);

  std::cout << "test";

  int counter = 0;
  int examinationsNumber = 100;
  std::vector<std::chrono::duration<double>> timer;
  int size = 4;
  for (int i = 0; i < examinationsNumber; i++)
  {
    // create an hpacket with the random data
    std::vector<uint8_t> cs1 = generateRandomVector(size);
    hpacket p1(cs1, MACs, key1, private_key1, 1);

    // start the timer
    auto start = std::chrono::high_resolution_clock::now();
    // check the integrity
    p1.macVerifier();
    p1.signVerifier();
    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    // stop-start
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    timer.push_back(duration);
  }
  std::cout << "here";

  return 0;
};
//};
