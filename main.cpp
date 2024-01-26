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
// #include "headers/node.h"
// #include "headers/tree_alg.h"
#include <chrono>
#include <fstream>
#include <cmath>
#include <set>
#include <numeric>

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
};
std::vector<int> generateRandomARvector(int ARsize)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(1, 100);

  std::vector<int> ARvector(ARsize);

  for (int i = 0; i < ARsize; ++i)
  {
    ARvector[i] = dis(gen);
  };

  return ARvector;
};
//////// Generate Polluted Packet Index ////////////////
std::vector<int> pollutionIndexselector(int gnrtnSize, int NumberOfPollution,std::vector<int> _probabilities)
{
  // Specify the probabilities for each index
  //     std::cout << "here";

  int a = std::accumulate(_probabilities.begin(), _probabilities.end(), 0);
  std::vector<int> transform_probability(a, 0);
  int cnt = 0;
  for (int k = 0; k < _probabilities.size(); k++)
  {
    for (int m = 0; m < _probabilities[k]; m++)
    {
      transform_probability[cnt] = k;
      cnt++;
    };
  };
  //   std::cout << "here";

  // Create a random number generator (use std::random_device for better randomness)
  std::random_device rd;
  std::mt19937 generator(std::random_device{}());
  std::vector<int> pollutedPacketIndex(NumberOfPollution, 0);
  // Create a discrete distribution based on the probabilities
  std::uniform_int_distribution<int> distribution(0, transform_probability.size() - 1);
  // Sample an index based on probabilities
  std::set<int> uniqueValues;

  for (int i = 0; i < NumberOfPollution; i++)
  {
    int generatedValue = 0;

    do
    {
      generatedValue = transform_probability[distribution(generator)];
    } while (!uniqueValues.insert(generatedValue).second); // Continue generating if the value is not unique

    pollutedPacketIndex[i] = generatedValue;
  };
  return pollutedPacketIndex;
};
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
};
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

  // std::vector<std::vector<uint8_t>> key1 = {{2, 3, 1, 4, 2}, {5, 8, 1, 19, 35}, {75, 15, 7, 9, 58}, {175, 156, 47, 91, 67}, {64, 23, 43, 56, 198}, {92, 37, 15, 42, 26}, {53, 82, 11, 91, 53}, {57, 51, 8, 6, 85}, {17, 106, 74, 19, 76}, {60, 25, 45, 55, 178}};

  // std::vector<uint8_t> private_key1 = {2, 4, 13, 111, 68};
  //  std::vector<uint8_t> private_key2 = {1, 7, 123, 8};
  //  std::vector<uint8_t> private_key3 = {55, 58, 71};
  // std::vector<uint8_t> private_key = {5, 4, 3, 2, 1, 67, 89, 12, 23, 33, 234};

  std::vector<std::vector<uint8_t>> MACs;
  std::vector<std::vector<uint8_t>> verified_symbols;

  // uint8_t currentsign;
  //  uint8_t c_sign;
  // uint8_t sign;
  //  Create a directed graph with 4 nodes
  //   Graph myGraph(4);
  //  std::vector<uint8_t> cs2 = {7, 4, 2};
  //  std::vector<std::vector<uint8_t>> key2 = {{2, 3, 1, 4},{2, 3, 1, 3},{1,2,3,4}};

  //  hpacket p2(cs2, MACs, key2, private_key2, 3);

  // std::vector<uint8_t> cs3 = {8, 4};
  // std::vector<std::vector<uint8_t>> key3 = {{2, 3, 1},{2, 1, 7}};
  // hpacket p3(cs3, MACs, key3, private_key3, 2);

  // std::cout << "test";

  int counter = 0;
  int examinationsNumber = 1 * 99999;
  std::vector<double> sum;
  std::vector<std::chrono::duration<double>> timer;
  std::vector<std::chrono::duration<double>> timerCombiner;
  int minPacketSize = 3;
  int maxPacketSize = 14;
  int packetStep = 2;
  int minMACSize = 2;
  int maxMACSize = 23;
  int MACStep = 5;
  int NumberOfLayers = 4;

  int NUmberOfLeaves = 2;
  double generationSize_double = std::pow(NUmberOfLeaves, NumberOfLayers - 1);
  int generationSize = static_cast<int>(generationSize_double);
  int NumberOfPOllutedPackets = 3;
  int packetSize = 5;
  int MACNumber = 3;
  std::vector<int> simple_counter(examinationsNumber, 0);
  std::vector<int> tree_counter(examinationsNumber, 0);
  std::vector<int> ar_tree_counter(examinationsNumber, 0);

  // change the file name based on the given steup

  // std::string filename = "./Results/Packet" + std::to_string(minPacketSize) + "-" + std::to_string(maxPacketSize);
  // filename += "MAC" + std::to_string(minMACSize) + "-" + std::to_string(maxMACSize);
  // filename += "ExNum:" + std::to_string(examinationsNumber);
  // filename += ".txt";

  // for (int packetSize = minPacketSize; packetSize < maxPacketSize; packetSize += packetStep)
  //{
  for (int pollutionNumber = 1; pollutionNumber < 5; pollutionNumber++)
  {
    // for (int MACNumber = minMACSize; MACNumber < maxMACSize; MACNumber += MACStep)
    // {
    std::string filename = "./Results/Packet" + std::to_string(packetSize);
    filename += "Generation Size" + std::to_string(generationSize);
    filename += "Number of Polluted Packets" + std::to_string(pollutionNumber);
    filename += "ExNum:" + std::to_string(examinationsNumber);
    filename += ".txt";

    std::ofstream outputFile(filename, std::ios::app);

    if (!outputFile.is_open())
    {
      std::cerr << "Error opening the file!" << std::endl;
      return 1;
    };
    // generate the keys for MACs based on the packet size
    std::vector<std::vector<uint8_t>> key1(MACNumber, std::vector<uint8_t>(packetSize + 1, 0));
    for (int i = 0; i < MACNumber; i++)
    {
      std::vector<uint8_t> newKey = generateRandomVector(packetSize + 1);
      key1[i] = newKey;
    };
    // generate the keys for sign based on the packet size
    // std::vector<uint8_t> private_key;
    // for (int i = 0; i < MACNumber; i++)
    //{
    std::vector<uint8_t> private_key = generateRandomVector(MACNumber + 1);
    // private_key.push_back(newPrivateKey);
    //};

    // int total_time=0;
    // timer.clear();
    // timerCombiner.clear();
    for (int i = 0; i < examinationsNumber; i++)
    {
      if (i % 10000 == 0)
      {
        std::cout << "i :" << i << std::endl;
      }
      std::vector<int> probabilities = generateRandomARvector(generationSize);
      std::vector<std::vector<uint8_t>> receivedPackets(generationSize, std::vector<uint8_t>(packetSize, 0));
      // std::cout << "here";

      std::vector<uint8_t> coefficientVector = generateRandomVector(generationSize);
      // create an hpacket with the random data
      for (int j = 0; j < generationSize; j++)
      {

        std::vector<uint8_t> cs1 = generateRandomVector(packetSize);
        receivedPackets[j] = cs1;
      };

      hpacket p1(receivedPackets, MACs, key1, private_key, MACNumber, coefficientVector);
      // TreeGenerator x1(receivedPackets);
      int a = packetSize + MACNumber + 1;
      std::vector<std::vector<uint8_t>> verifierSymbols(generationSize, std::vector<uint8_t>(a, 0));

      // start the timer
      // auto start = std::chrono::high_resolution_clock::now();

      // check the integrity
      p1.macCalculator(); // fixed
      p1.signCalculator();
      // p1.packetAppender(receivedPackets);                   // fixed
      verifierSymbols = p1.packetAppender(receivedPackets); // fixed
      std::vector<int> pIv = pollutionIndexselector(generationSize, pollutionNumber,probabilities);
      //    p1.macVerifier(verifierSymbols[0]);      //fixed
      // std::cout << "here";

      // p1.signVerifier(verifierSymbols[0]);     //fixed
      // std::cout << "here";

      // std::cout << "here";

      std::vector<std::vector<uint8_t>> pollutedVerifierSymbols = p1.pollutionGeneration(verifierSymbols, pIv);
      std::vector<std::vector<std::vector<uint8_t>>> verificationTree = p1.treeGenerator(pollutedVerifierSymbols, NumberOfLayers, NUmberOfLeaves, verifierSymbols[0].size()); /// fixed
      // std::cout << "here";
      // std::cout << "The round starts here";
      tree_counter[i] = p1.treeVerifier(verificationTree, NumberOfLayers, NUmberOfLeaves);
      //  std::cout << "TreeVerifier done";
      simple_counter[i] = p1.simpleVerifier(verifierSymbols);                                                           // fixed
                                                                                                                        // std::cout << "simpleverifier done";
      ar_tree_counter[i] = p1.arTreeVerifier(verificationTree, probabilities, NumberOfLayers); // AR based tree algorithm done !!
      //  std::cout << "ARTreeVerifier done";

      //  std::cout << "here";
    };
    // stop the timer
    // auto end = std::chrono::high_resolution_clock::now();

    // print the result and put it in the file
    //   outputFile << "PacketSize:" << packetSize << "-"
    //            << "MACSize:" << MACNumber << "-Result:" << sum_size / examinationsNumber << std::endl;
    // outputFile << "PacketSize:" << packetSize << "-"
    //         << "MACSize:" << MACNumber << "-ResultCombiner:" << sum_sizeCombiner / examinationsNumber << std::endl;
    // outputFile.flush();
    //  print the result and put it in the file
    int avg_simple = std::accumulate(simple_counter.begin(), simple_counter.end(), 0) / examinationsNumber;
    int avg_tree = std::accumulate(tree_counter.begin(), tree_counter.end(), 0) / examinationsNumber;
    int avg_ar_tree = std::accumulate(ar_tree_counter.begin(), ar_tree_counter.end(), 0) / examinationsNumber;

    outputFile << "PacketSize:" << packetSize << "-"
               << "GenerationSize:" << generationSize << "Pollution Number" << pollutionNumber << "-SImple Ver Result:" << avg_simple << std::endl;
    outputFile << "PacketSize:" << packetSize << "-"
               << "GenerationSize:" << generationSize << "Pollution Number" << pollutionNumber << "-Tree VerResult:" << avg_tree << std::endl;
    outputFile << "PacketSize:" << packetSize << "-"
               << "GenerationSize:" << generationSize << "Pollution Number" << pollutionNumber << "-AR Tree Ver Result:" << avg_ar_tree << std::endl;
    outputFile.flush();
    //}
    outputFile << std::endl;
    outputFile.flush();
  };
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

  // std::cout << "here";

  //      std::cout << "here";

  return 0;
};
