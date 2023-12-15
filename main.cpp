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
#include "headers/node.h"
#include <chrono>
#include <fstream>

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
// Generate coefficient vector
std::vector<uint8_t> generateChannelVector(int generationsize)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dis(0, 255);

  std::vector<uint8_t> coefficientVector(generationsize);

  for (int i = 0; i < generationsize; ++i)
  {
    coefficientVector[i] = dis(gen);
  }

  return coefficientVector;
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

  //std::vector<std::vector<uint8_t>> key1 = {{2, 3, 1, 4, 2}, {5, 8, 1, 19, 35}, {75, 15, 7, 9, 58}, {175, 156, 47, 91, 67}, {64, 23, 43, 56, 198}, {92, 37, 15, 42, 26}, {53, 82, 11, 91, 53}, {57, 51, 8, 6, 85}, {17, 106, 74, 19, 76}, {60, 25, 45, 55, 178}};

  //std::vector<uint8_t> private_key1 = {2, 4, 13, 111, 68};
  // std::vector<uint8_t> private_key2 = {1, 7, 123, 8};
  // std::vector<uint8_t> private_key3 = {55, 58, 71};
  //std::vector<uint8_t> private_key = {5, 4, 3, 2, 1, 67, 89, 12, 23, 33, 234};

  std::vector<uint8_t> MACs;
  uint8_t currentsign;
  // uint8_t c_sign;
  uint8_t sign;
  // Create a directed graph with 4 nodes
  //  Graph myGraph(4);
  // std::vector<uint8_t> cs2 = {7, 4, 2};
  // std::vector<std::vector<uint8_t>> key2 = {{2, 3, 1, 4},{2, 3, 1, 3},{1,2,3,4}};
  // hpacket p2(cs2, MACs, key2, private_key2, 3);

  // std::vector<uint8_t> cs3 = {8, 4};
  // std::vector<std::vector<uint8_t>> key3 = {{2, 3, 1},{2, 1, 7}};
  // hpacket p3(cs3, MACs, key3, private_key3, 2);

 // std::cout << "test";

  int counter = 0;
  int examinationsNumber = 1 * 10;
  std::vector<double> sum;
  std::vector<std::chrono::duration<double>> timer;
  std::vector<std::chrono::duration<double>> timerCombiner;
  int minPacketSize = 10;
  int maxPacketSize = 21;
  int packetStep = 2;
  int generationSize = 5;
  int minMACSize = 5;
  int maxMACSize = 21;
  int MACStep = 5;

  // change the file name based on the given steup

  std::string filename = "./Results/Packet" + std::to_string(minPacketSize) + "-" + std::to_string(maxPacketSize);
  filename += "MAC" + std::to_string(minMACSize) + "-" + std::to_string(maxMACSize);
  filename += "ExNum:" + std::to_string(examinationsNumber);
  filename += ".txt";

  std::ofstream outputFile(filename, std::ios::app);

  if (!outputFile.is_open())
  {
    std::cerr << "Error opening the file!" << std::endl;
    return 1;
  }

  for (int packetSize = minPacketSize; packetSize < maxPacketSize; packetSize += packetStep)
  {
    for (int MACNumber = minMACSize; MACNumber < maxMACSize; MACNumber += MACStep)
    {

      // generate the keys for MACs based on the packet size
      std::vector<std::vector<uint8_t>> key1;
      for (int i = 0; i < MACNumber; i++)
      {
        std::vector<uint8_t> newKey = generateRandomVector(packetSize + 1);
        key1.push_back(newKey);
      };
      // generate the keys for sign based on the packet size
      //std::vector<uint8_t> private_key;
      //for (int i = 0; i < MACNumber; i++)
      //{
       std::vector<uint8_t> private_key = generateRandomVector(MACNumber + 1);
        //private_key.push_back(newPrivateKey);
      //};

      // int total_time=0;
      timer.clear();
      timerCombiner.clear();
      for (int i = 0; i < examinationsNumber; i++)
      {
        std::vector<uint8_t> coefficientVector = generateRandomVector(generationSize);
        // create an hpacket with the random data
        std::vector<uint8_t> cs1 = generateRandomVector(packetSize);
        std::vector<uint8_t> cs2 = generateRandomVector(packetSize);


        hpacket p1(cs1, MACs, key1, private_key, MACNumber,coefficientVector);

        // start the timer
        auto start = std::chrono::high_resolution_clock::now();
        // check the integrity
        p1.macVerifier();
        p1.signVerifier();
        // stop the timer
        auto end = std::chrono::high_resolution_clock::now();
        // stop-start
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        timer.push_back(duration);

        // start the timer for combiner
        auto startCombiner = std::chrono::high_resolution_clock::now();
        // combine packets
        p1.packetCombiner();
        // stop the timer
        auto endCombiner = std::chrono::high_resolution_clock::now();
        // stop-start
        auto durationCombiner = std::chrono::duration_cast<std::chrono::nanoseconds>(endCombiner - startCombiner);
        timerCombiner.push_back(durationCombiner);
      }
      double sum_size = 0;
      for (const auto &entry : timer)
      {
        sum_size += entry.count(); // Adding each duration to the sum in milliseconds
      }
      sum.push_back(sum_size / examinationsNumber);
      //
      double sum_sizeCombiner = 0;
      for (const auto &entry : timerCombiner)
      {
        sum_sizeCombiner += entry.count(); // Adding each duration to the sum in milliseconds
      }
      sum.push_back(sum_sizeCombiner / examinationsNumber);
      // print the result and put it in the file
      outputFile << "PacketSize:" << packetSize << "-"
                 << "MACSize:" << MACNumber << "-Result:" << sum_size / examinationsNumber << std::endl;
      outputFile << "PacketSize:" << packetSize << "-"
                 << "MACSize:" << MACNumber << "-ResultCombiner:" << sum_sizeCombiner / examinationsNumber << std::endl;
      outputFile.flush();
    }
    outputFile << std::endl;
    outputFile.flush();
  }
// Set inputs for Node 0, Node 1, and Node 2 with different types of data
  //  NodeInput input0 = {{1, 2, 3}, {{4, 5}, {6, 7}}, 8};
  //  NodeInput input1 = {{9, 10}, {{11, 12}, {13, 14}}, 15};
 //   NodeInput input2 = {{16, 17, 18}, {{19, 20}, {21, 22}}, 23};

 //   myGraph.setInput(0, input0);
  //  myGraph.setInput(1, input1);
 //   myGraph.setInput(2, input2);

    // Add directed edges between nodes
  //  myGraph.addDirectedEdge(0, 1);
 //   myGraph.addDirectedEdge(0, 2);


    std::cout << "here";


  return 0;
};
//};
