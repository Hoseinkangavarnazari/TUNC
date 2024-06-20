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
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/property_map.hpp>

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
///////////////// Generate random AR vector //////////////////////////////////
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
///////////////// Initialize AR vector //////////////////////////////////

std::vector<int> initializeARvector(int ARsize)
{

  std::vector<int> initialARvector(ARsize);

  for (int i = 0; i < ARsize; ++i)
  {
    initialARvector[i] = 0;
    // initialARvector[i+(ARsize/2)]=0;
  };
  initialARvector[0] = 1;
  initialARvector[1] = 1;
  initialARvector[2] = 1;
  initialARvector[3] = 1;

  return initialARvector;
};

//////// Generate Polluted Packet Index ////////////////
std::vector<int> pollutionIndexselector(int gnrtnSize, int NumberOfPollution, std::vector<int> _probabilities)
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

std::vector<int> pollutionIndexselectorNEW(int gnrtnSize, int NumberOfPollution)
{
  // Specify the probabilities for each index
  //     std::cout << "here";

  //   std::cout << "here";

  // Create a random number generator (use std::random_device for better randomness)
  std::random_device rd;
  std::mt19937 generator(std::random_device{}());
  std::vector<int> pollutedPacketIndex(NumberOfPollution, 0);
  // Create a discrete distribution based on the probabilities
  std::uniform_int_distribution<int> distribution(0, gnrtnSize - 1);
  // Sample an index based on probabilities
  std::set<int> uniqueValues;

  for (int i = 0; i < NumberOfPollution; i++)
  {
    int generatedValue = 0;

    do
    {
      generatedValue = distribution(generator);
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
// Initialize random number generator
  std::random_device rd;
  std::mt19937 generator(std::random_device{}());

//----------------------------------------- Key Distribution Center------------------------------------------------------//
std::vector<std::vector<uint8_t>> keyDistributor(int _KeysetSize, std::vector<std::vector<uint8_t>> KeyPool)
{
 // std::cout << "here";

  std::vector<int> chosenSet;
  std::vector<std::vector<uint8_t>> assignedKeyset(_KeysetSize, std::vector<uint8_t>(KeyPool[0].size(),0));
  
  std::vector<int> chosenNumbers(_KeysetSize, 0);
  // Create a discrete distribution based on the probabilities
  std::uniform_int_distribution<int> distribution(0, KeyPool.size() - 1);
  // Sample an index based on probabilities
  std::set<int> uniqueValues;

  for (int i = 0; i < _KeysetSize; i++)
  {
    int generatedValue = 0;

    do
    {
      generatedValue = distribution(generator);
    } while (!uniqueValues.insert(generatedValue).second); // Continue generating if the value is not unique

    chosenNumbers[i] = generatedValue;
  };
  for (int i = 0; i < _KeysetSize; i++)
  {

    assignedKeyset[i] = KeyPool[chosenNumbers[i]];
  };
  return assignedKeyset;
};

//------------------------------------------------------------------------------------------------------------------------//
// int mainOLD()           ///////////MAIN START/////////////////
// {

//   // // std::vector<std::vector<uint8_t>> test_mac;
//   // hpacket hp(hcodedSymbol, publickeyset, publickeyset.size());
//   // // test_mac = macCalculator(hcodedSymbol, publickeyset, 5);
//   // FieldSize fieldSize = BinaryC;
//   // int dataSizeInBytes = generationSize * symbolSize;
//   // std::vector<uint8_t> sampleData = randomDataGenerator(dataSizeInBytes, fieldSize);
//   // std::vector<std::vector<uint8_t>> test_mac = hp.macCalculator(hcodedSymbol, publickeyset, publickeyset.size());
//   // rlnc_encoder encoder(generationSize, symbolSize, fieldSize);
//   // rlnc_decoder decoder(generationSize, symbolSize, fieldSize);
//   // bool res = encoder.setSymbol(sampleData);
//   // // while
//   // std::vector<packet> allPackets;
//   // int packetCount = 0;
//   // while (packetCount != generationSize)
//   // {
//   //   // create a coded packet
//   //   // First, generate a random coefficient
//   //   std::vector<uint8_t> tempCoeff;
//   //   encoder.randomCoeffGenerator(tempCoeff);
//   //   // Second, encode using the coeff
//   //   std::vector<uint8_t> codedSybmols = encoder.encode_rlnc(tempCoeff);
//   //   // Third, create a packet object
//   //   // packet p(tempCoeff, codedSybmols,packetCount);

//   //   // give the packet to the decoder
//   //   // allPackets.push_back(p);
//   //   // decoder.consumeCodedPacket(p);
//   //   // packetCount++;
//   // }
//   // decoder.decode();

//   // std::vector<std::vector<uint8_t>> key1 = {{2, 3, 1, 4, 2}, {5, 8, 1, 19, 35}, {75, 15, 7, 9, 58}, {175, 156, 47, 91, 67}, {64, 23, 43, 56, 198}, {92, 37, 15, 42, 26}, {53, 82, 11, 91, 53}, {57, 51, 8, 6, 85}, {17, 106, 74, 19, 76}, {60, 25, 45, 55, 178}};

//   // std::vector<uint8_t> private_key1 = {2, 4, 13, 111, 68};
//   //  std::vector<uint8_t> private_key2 = {1, 7, 123, 8};
//   //  std::vector<uint8_t> private_key3 = {55, 58, 71};
//   // std::vector<uint8_t> private_key = {5, 4, 3, 2, 1, 67, 89, 12, 23, 33, 234};

//   std::vector<std::vector<uint8_t>> MACs;
//   std::vector<std::vector<uint8_t>> verified_symbols;

//   // uint8_t currentsign;
//   //  uint8_t c_sign;
//   // uint8_t sign;
//   //  Create a directed graph with 4 nodes
//   //   Graph myGraph(4);
//   //  std::vector<uint8_t> cs2 = {7, 4, 2};
//   //  std::vector<std::vector<uint8_t>> key2 = {{2, 3, 1, 4},{2, 3, 1, 3},{1,2,3,4}};

//   //  hpacket p2(cs2, MACs, key2, private_key2, 3);

//   // std::vector<uint8_t> cs3 = {8, 4};
//   // std::vector<std::vector<uint8_t>> key3 = {{2, 3, 1},{2, 1, 7}};
//   // hpacket p3(cs3, MACs, key3, private_key3, 2);

//   // std::cout << "test";

//   int counter = 0;
//   int examinationsNumber = 1 * 1000;
//   std::vector<double> sum;
//   std::vector<std::chrono::duration<double>> timer;
//   std::vector<std::chrono::duration<double>> timer_simple_verifier;
//   std::vector<std::chrono::duration<double>> timer_tree_verifier;
//   std::vector<std::chrono::duration<double>> timer_ar_verifier;
//   std::vector<std::chrono::duration<double>> mean_timer_simple_verifier;
//   std::vector<std::chrono::duration<double>> mean_timer_tree_verifier;
//   std::vector<std::chrono::duration<double>> mean_timer_ar_verifier;
//   std::vector<std::chrono::duration<double>> timer_summation;
//   std::vector<std::chrono::duration<double>> timer_verification;
//   std::vector<std::chrono::duration<double>> timer_multiplication;
//   std::vector<std::chrono::duration<double>> mean_timer_multiplication;
//   std::vector<std::chrono::duration<double>> timerCombiner;
//   std::vector<std::chrono::duration<double>> mean_timer_summation;
//   int minPacketSize = 100;
//   int maxPacketSize = 501;
//   int packetStep = 10;
//   int minMACSize = 2;
//   int maxMACSize = 23;
//   int MACStep = 5;
//   int NumberOfLayers = 8;

//   int NUmberOfLeaves = 2;
//   double generationSize_double = std::pow(NUmberOfLeaves, NumberOfLayers - 1);
//   int generationSize = static_cast<int>(generationSize_double);
//   int NumberOfPOllutedPackets = 3;
//   // int packetSize = 5;
//   int MACNumber = 3;
//   int pollutionNumber = 2;
//   std::vector<int> simple_counter(examinationsNumber, 0);
//   std::vector<int> tree_counter(examinationsNumber, 0);
//   std::vector<int> ar_tree_counter(examinationsNumber, 0);

//   // change the file name based on the given steup

//   // std::string filename = "./Results/Packet" + std::to_string(minPacketSize) + "-" + std::to_string(maxPacketSize);
//   // filename += "MAC" + std::to_string(minMACSize) + "-" + std::to_string(maxMACSize);
//   // filename += "ExNum:" + std::to_string(examinationsNumber);
//   // filename += ".txt";
//   int cnt = 0;
//   for (int packetSize = minPacketSize; packetSize < maxPacketSize; packetSize += packetStep)
//   {
//     // for (int pollutionNumber = 1; pollutionNumber < 5; pollutionNumber++)
//     //{
//     //  for (int MACNumber = minMACSize; MACNumber < maxMACSize; MACNumber += MACStep)
//     //  {

//     std::string filename = "./Results/Packet" + std::to_string(packetSize);
//     filename += "Generation Size" + std::to_string(generationSize);
//     filename += "Number of Polluted Packets" + std::to_string(pollutionNumber);
//     filename += "ExNum:" + std::to_string(examinationsNumber);
//     filename += ".txt";

//     std::ofstream outputFile(filename, std::ios::app);

//     if (!outputFile.is_open())
//     {
//       std::cerr << "Error opening the file!" << std::endl;
//       return 1;
//     };
//     // generate the keys for MACs based on the packet size
//     std::vector<std::vector<uint8_t>> key1(MACNumber, std::vector<uint8_t>(packetSize + 1, 0));
//     for (int i = 0; i < MACNumber; i++)
//     {
//       std::vector<uint8_t> newKey = generateRandomVector(packetSize + 1);
//       key1[i] = newKey;
//     };
//     // generate the keys for sign based on the packet size
//     // std::vector<uint8_t> private_key;
//     // for (int i = 0; i < MACNumber; i++)
//     //{
//     std::vector<uint8_t> private_key = generateRandomVector(MACNumber + 1);
//     // private_key.push_back(newPrivateKey);
//     //};

//     // int total_time=0;
//     // timer.clear();
//     // timer_multiplication.clear();
//     // timer_summation.clear();
//     // timerCombiner.clear();
//     for (int i = 0; i < examinationsNumber; i++)
//     {
//       if (i % 100 == 0)
//       {
//         std::cout << "i :" << i << std::endl;
//       }
//       //      std::vector<int> probabilities = generateRandomARvector(generationSize);
//       std::vector<int> probabilities = initializeARvector(generationSize);
//       //        std::cout << "here";

//       std::vector<std::vector<uint8_t>> receivedPackets(generationSize, std::vector<uint8_t>(packetSize, 0));
//       // std::cout << "here";

//       std::vector<uint8_t> coefficientVector = generateRandomVector(generationSize);
//       // create an hpacket with the random data
//       for (int j = 0; j < generationSize; j++)
//       {

//         std::vector<uint8_t> cs1 = generateRandomVector(packetSize);
//         receivedPackets[j] = cs1;
//       };

//       hpacket p1(receivedPackets, MACs, key1, private_key, MACNumber, coefficientVector);
//       // TreeGenerator x1(receivedPackets);
//       int a = packetSize + MACNumber + 1;
//       std::vector<std::vector<uint8_t>> verifierSymbols(generationSize, std::vector<uint8_t>(a, 0));

//       // start the timer
//       // auto start = std::chrono::high_resolution_clock::now();

//       // check the integrity
//       p1.macCalculator(); // fixed
//       p1.signCalculator();
//       // p1.packetAppender(receivedPackets);                   // fixed
//       verifierSymbols = p1.packetAppender(receivedPackets); // fixed
//       std::vector<int> pIv = pollutionIndexselector(generationSize, pollutionNumber, probabilities);
//       //    std::cout << "here";

//       //    p1.macVerifier(verifierSymbols[0]);      //fixed
//       // std::cout << "here";

//       // p1.signVerifier(verifierSymbols[0]);     //fixed
//       // std::cout << "here";
//       // std::cout << "here";
//       ///////////////////////////////////////  POLLUTION GENERATOR ////////////////////////////
//       std::vector<std::vector<uint8_t>> pollutedVerifierSymbols = p1.pollutionGeneration(verifierSymbols, pIv);
//       std::vector<std::vector<std::vector<uint8_t>>> verificationTree = p1.treeGenerator(pollutedVerifierSymbols, NumberOfLayers, NUmberOfLeaves, verifierSymbols[0].size()); /// fixed

//       ///////////////////////////////////////  TREE GENERATOR & VERIFIER and TIME MEASUREMENT ///////////////////////////
//       // activate again  auto start_tree_verifier = std::chrono::high_resolution_clock::now();
//       // std::cout << "here";
//       // std::cout << "The round starts here";
//       tree_counter[i] = p1.treeVerifier(verificationTree, NumberOfLayers, NUmberOfLeaves);
//       // activate again  auto end_tree_verifier = std::chrono::high_resolution_clock::now();
//       // activate again  auto duration_tree_verifier = std::chrono::duration_cast<std::chrono::microseconds>(end_tree_verifier- start_tree_verifier);
//       // activate again  timer_tree_verifier.push_back(duration_tree_verifier);
//       //  std::cout << "TreeVerifier done";
//       ///////////////////////////////////////  SIMPLE VERIFIER and TIME MEASUREMENT ///////////////////////////
//       // activate again auto start_simple_verifier = std::chrono::high_resolution_clock::now();
//       simple_counter[i] = p1.simpleVerifier(verifierSymbols); // fixed
//       // activate again  auto end_simple_verifier = std::chrono::high_resolution_clock::now();
//       // activate again  auto duration_simple_verifier = std::chrono::duration_cast<std::chrono::microseconds>(end_simple_verifier- start_simple_verifier);
//       // activate again  timer_simple_verifier.push_back(duration_simple_verifier);
//       ///////////////////////////////////////  TREE GENERATOR & AR VERIFIER and TIME MEASUREMENT ///////////////////////////
//       // activate again auto start_ar_verifier = std::chrono::high_resolution_clock::now();
//       //  std::vector<std::vector<std::vector<uint8_t>>> verificationTree_ar = p1.treeGenerator(pollutedVerifierSymbols, NumberOfLayers, NUmberOfLeaves, verifierSymbols[0].size()); /// fixed
//       // std::cout << "simpleverifier done";
//       ar_tree_counter[i] = p1.arTreeVerifier(verificationTree, probabilities, NumberOfLayers); // AR based tree algorithm done !!
//       // activate again  auto end_ar_verifier = std::chrono::high_resolution_clock::now();
//       // activate again auto duration_ar_verifier = std::chrono::duration_cast<std::chrono::microseconds>(end_ar_verifier- start_ar_verifier);
//       // activate again  timer_ar_verifier.push_back(duration_ar_verifier);
//       //  std::cout << "ARTreeVerifier done";

//       std::vector<uint8_t> rdnm1 = generateRandomVector(packetSize);
//       std::vector<uint8_t> rdnm2 = generateRandomVector(packetSize);
//       uint8_t rndmnmbr = 5;
//       ///////////////////////////// timer for single addition //////////////////////////////////
//       // activate again   auto start_combiner = std::chrono::high_resolution_clock::now();
//       p1.randomCombiner(rdnm1, rdnm2);
//       // activate again  auto end_combiner = std::chrono::high_resolution_clock::now();
//       // activate again  auto duration_combiner = std::chrono::duration_cast<std::chrono::microseconds>(end_combiner - start_combiner);
//       // activate again  timer_summation.push_back(duration_combiner);
//       ///////////////////////////// timer for single verification //////////////////////////////////
//       // activate again  auto start_verifier = std::chrono::high_resolution_clock::now();
//       p1.macVerifier(verifierSymbols[0]);
//       p1.signVerifier(verifierSymbols[0]);
//       // activate again  auto end_verifier = std::chrono::high_resolution_clock::now();
//       // activate again  auto duration_verifier = std::chrono::duration_cast<std::chrono::microseconds>(end_verifier - start_verifier);
//       // activate again  timer_verification.push_back(duration_verifier);
//       ///////////////////////////// timer for single multiplication //////////////////////////////////
//       // activate again  auto start_multiplier = std::chrono::high_resolution_clock::now();
//       p1.randomMultiplier(rndmnmbr, rdnm1);
//       // activate again  auto end_multiplier = std::chrono::high_resolution_clock::now();
//       // activate again  auto duration_multiplier = std::chrono::duration_cast<std::chrono::microseconds>(end_multiplier - start_multiplier);
//       // activate again timer_multiplication.push_back(duration_multiplier);
//     };
//     // stop the timer
//     // auto end = std::chrono::high_resolution_clock::now();

//     // print the result and put it in the file
//     //   outputFile << "PacketSize:" << packetSize << "-"
//     //            << "MACSize:" << MACNumber << "-Result:" << sum_size / examinationsNumber << std::endl;
//     // outputFile << "PacketSize:" << packetSize << "-"
//     //         << "MACSize:" << MACNumber << "-ResultCombiner:" << sum_sizeCombiner / examinationsNumber << std::endl;
//     // outputFile.flush();
//     //  print the result and put it in the file
//     ////////////////////////// TAKING MEAN VALUES FOR MEASUREMENTS  ////////////////////////////////////
//     //////// combination
//     // activate again   auto totalDuration_sum = std::accumulate(timer_summation.begin(), timer_summation.end(), std::chrono::duration<double>(0));
//     // activate again  auto sum_mean =totalDuration_sum/ examinationsNumber;
//     //////// verification
//     // activate again  auto totalDuration_verification = std::accumulate(timer_verification.begin(), timer_verification.end(), std::chrono::duration<double>(0));
//     // activate again  auto verification_mean =totalDuration_verification/ examinationsNumber;
//     //   mean_timer_summation[cnt]= sum_mean;
//     //////// multiplication
//     // activate again  auto totalDuration_multiply=std::accumulate(timer_multiplication.begin(), timer_multiplication.end(), std::chrono::duration<double>(0)) ;
//     // activate again  auto multip_mean = totalDuration_multiply/examinationsNumber;
//     ////////  simple verifier
//     // activate again  auto totalDuration_simple = std::accumulate(timer_simple_verifier.begin(), timer_simple_verifier.end(), std::chrono::duration<double>(0));
//     // activate again  auto simple_verifier_mean =totalDuration_simple/ examinationsNumber;
//     ////////  tree verifier
//     // activate again  auto totalDuration_tree = std::accumulate(timer_tree_verifier.begin(), timer_tree_verifier.end(), std::chrono::duration<double>(0));
//     // activate again  auto tree_verifier_mean =totalDuration_tree/ examinationsNumber;
//     ////////  ar verifier
//     // activate again  auto totalDuration_ar = std::accumulate(timer_ar_verifier.begin(), timer_ar_verifier.end(), std::chrono::duration<double>(0));
//     // activate again  auto ar_verifier_mean =totalDuration_ar/ examinationsNumber;

//     // mean_timer_multiplication[cnt] = multip_mean;
//     int avg_simple = std::accumulate(simple_counter.begin(), simple_counter.end(), 0) / examinationsNumber;
//     int avg_tree = std::accumulate(tree_counter.begin(), tree_counter.end(), 0) / examinationsNumber;
//     int avg_ar_tree = std::accumulate(ar_tree_counter.begin(), ar_tree_counter.end(), 0) / examinationsNumber;
//     cnt++;

//     std::cout << "here";

//     outputFile << "PacketSize:" << packetSize << "-"
//                << "GenerationSize:" << generationSize << "Pollution Number" << pollutionNumber << "-SImple Ver Check Number:" << avg_simple << std::endl;

//     outputFile << "PacketSize:" << packetSize << "-"
//                << "GenerationSize:" << generationSize << "Pollution Number" << pollutionNumber << "-Tree VerResult:" << avg_tree << std::endl;
//     outputFile << "PacketSize:" << packetSize << "-"
//                << "GenerationSize:" << generationSize << "Pollution Number" << pollutionNumber << "-AR Tree Ver Result:" << avg_ar_tree << std::endl;
//     outputFile.flush();
//     //}
//     outputFile << std::endl;
//     outputFile.flush();
//   };
//   // Set inputs for Node 0, Node 1, and Node 2 with different types of data
//   //  NodeInput input0 = {{1, 2, 3}, {{4, 5}, {6, 7}}, 8};
//   //  NodeInput input1 = {{9, 10}, {{11, 12}, {13, 14}}, 15};
//   //   NodeInput input2 = {{16, 17, 18}, {{19, 20}, {21, 22}}, 23};

//   //   myGraph.setInput(0, input0);
//   //  myGraph.setInput(1, input1);
//   //   myGraph.setInput(2, input2);

//   // Add directed edges between nodes
//   //  myGraph.addDirectedEdge(0, 1);
//   //   myGraph.addDirectedEdge(0, 2);

//   // std::cout << "here";

//   //      std::cout << "here";

//   return 0;
// };          /////////   MAIN END ///////

// .............................................................................................

struct VertexProperties
{
  int healthyReceived = 0;
  int pollutedReceived = 0;
  int pollutedDropped = 0;
  int totalNodeSend = 0;
  int falsePositiveEvents = 0;
  int checkNumber = 0;
  int buffercounter = 0;
  int numPaths = 0;
  std::vector<std::vector<uint8_t>> keySet;
  std::vector<std::vector<uint8_t>> receivedDataPackets;
  std::vector<std::vector<uint8_t>> nonSourceNodeBuffer;
  std::vector<std::vector<uint8_t>> nonSourceNodeRLNC;
  std::vector<std::vector<uint8_t>> SourceNodeOutput;
  std::string type = "";
  double attackProbability = 0.05;
  std::vector<uint8_t> output;
  std::vector<uint8_t> input;
  std::vector<int> arVectorForPackets;
  std::vector<int> path_index_vector;
};
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexProperties> Graph;

void simulation(Graph _topology, std::vector<std::vector<Graph::vertex_descriptor>> path_list, int _GG, int _G, int _fieldSize, int _packetSize, int _keypoolSize, int _keysetSize, int Number_of_Pollution, int _bufferSize);
// ........................................................
int main()
{
  // generate topology
  Graph g;

  // Add vertices to the graph and assign properties
  auto v0 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Source Node
  auto v1 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Intermediate_1 Node
  auto v2 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Compromised Node
  auto v3 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Intermediate_2 Node
  auto v4 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Destination Node

  // Create a connection between Nodes
  boost::add_edge(v0, v1, g); // From Source to Intermediate_1
  boost::add_edge(v0, v3, g); // From Source to Intermediate_2
  boost::add_edge(v0, v2, g); // From Source to Adversary
  boost::add_edge(v1, v2, g); // From Intermediate_1 to Adversary
  boost::add_edge(v2, v3, g); // From Adversary to Intermediate
  boost::add_edge(v1, v3, g); // From Intermediate_1 to Intermediate_2
  boost::add_edge(v3, v4, g); // From  Intermediate_2 to Destination
  boost::add_edge(v1, v4, g); // From  Intermediate_1 to Destination

  // Define paths
  std::vector<Graph::vertex_descriptor> path_1 = {v0, v1, v4};
  std::vector<Graph::vertex_descriptor> path_2 = {v0, v1, v3, v2, v4};
  std::vector<Graph::vertex_descriptor> path_3 = {v0, v1, v3, v4};
  std::vector<Graph::vertex_descriptor> path_4 = {v0, v3, v4};
  std::vector<Graph::vertex_descriptor> path_5 = {v0, v2, v4};
  std::vector<Graph::vertex_descriptor> path_6 = {v0, v1, v2, v4};
  std::vector<Graph::vertex_descriptor> path_7 = {v0, v3, v2, v4};
  std::vector<Graph::vertex_descriptor> path_8 = {v0, v3, v1, v2, v4};
  // // Count the number of paths
  //   numPaths += !path_1.empty();
  //   numPaths += !path_2.empty();
  //   numPaths += !path_3.empty();
  //   numPaths += !path_4.empty();
  //   numPaths += !path_5.empty();

  std::vector<std::vector<Graph::vertex_descriptor>> path_list;
  path_list.push_back(path_1);
  path_list.push_back(path_2);
  path_list.push_back(path_3);
  path_list.push_back(path_4);
  path_list.push_back(path_5);
  path_list.push_back(path_6);
  path_list.push_back(path_7);
  path_list.push_back(path_8);

  // set the main simulation parameters
  int _GG = 10; // number of generations
  int _G = 64; // generationsize
  int _fieldSize = 256;
  int _packetSize = 5;
  int _keypoolSize = 8;
  int _keysetSize = 4;
  int Number_of_Pollution = 1;
  int _bufferSize = _G;

  // Access and manipulate vertex properties
  // Source Node
  g[v0].pollutedReceived = 0;
  g[v0].pollutedDropped = 0;
  g[v0].type = "Source";
  g[v0].totalNodeSend = 0;
  // Intermediate_1 Node
  g[v1].pollutedReceived = 0;
  g[v1].pollutedDropped = 0;
  g[v1].type = "Intermediate";
  g[v1].totalNodeSend = 0;
  g[v1].checkNumber = 0;
  // Adversary Node
  g[v2].pollutedReceived = 0;
  g[v2].pollutedDropped = 0;
  g[v2].type = "Adversary";
  g[v2].totalNodeSend = 0;
  g[v2].checkNumber = 0;
  // Intermediate_2 Node
  g[v3].pollutedReceived = 0;
  g[v3].pollutedDropped = 0;
  g[v3].type = "Intermediate";
  g[v3].totalNodeSend = 0;
  g[v3].checkNumber = 0;
  // Destination Node
  g[v4].pollutedReceived = 0;
  g[v4].pollutedDropped = 0;
  g[v4].type = "Destination";
  g[v4].totalNodeSend = 0;
  g[v4].checkNumber = 0;
  g[v4].path_index_vector = std::vector<int>(_bufferSize,0);
  g[v4].arVectorForPackets = std::vector<int>(_bufferSize,0);


  simulation(g, path_list, _GG, _G, _fieldSize, _packetSize, _keypoolSize, _keysetSize, Number_of_Pollution, _bufferSize);
}

void simulation(Graph _topology, std::vector<std::vector<Graph::vertex_descriptor>> path_list, int _GG, int _G, int _fieldSize, int _packetSize, int _keypoolSize, int _keysetSize, int Number_of_Pollution, int _bufferSize)
{
  // Get the number of nodes in the graph
  std::size_t numVertices = boost::num_vertices(_topology);

  // Convert the number of vertices to an int
  int number_of_nonsource_nodes = static_cast<int>(numVertices) - 1;

  // Initialize the input of source node
  std::vector<uint8_t> zeroInputVector(std::vector<uint8_t>(_packetSize, 0));

  // Initialize the buffer of non-source nodes
  std::vector<std::vector<uint8_t>> zeroBufferMatrice(_bufferSize, std::vector<uint8_t>(_packetSize + _keypoolSize + 1, 0));

  // Initialize the input of non-source node
  std::vector<uint8_t> zeroVector(std::vector<uint8_t>(_packetSize + _keypoolSize + 1, 0));

  // Initialize the AR vectors at non-source nodes
  std::vector<int> zeroARVector(std::vector<int>(_bufferSize, 0)); // 5 is the number of paths

  std::vector<int> ARvector(std::vector<int>(8, 1)); // Size of AR vector will be updated

  std::vector<std::vector<int>> result_vector_treeVerifier(_GG,std::vector<int>(2, 0)); // Size of AR vector will be updated

  std::vector<std::vector<int>> result_vector_arVerifier(_GG,std::vector<int>(2, 0)); // Size of AR vector will be updated

  std::vector<std::vector<int>> result_vector_simpleVerifier(_GG,std::vector<int>(2, 0)); // Size of AR vector will be updated



  std::srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  

  //----------------------------------------------  Start Each Generation  -------------------------------------------------------//
  for (int generationIndex = 0; generationIndex < _GG; generationIndex++)
  {
    bool flag = true;
    // Initialize buffers of each nonsource nodes in each generation
    for (int nodeIndex = 0; nodeIndex < number_of_nonsource_nodes; nodeIndex++)
    {
      _topology[nodeIndex+1].nonSourceNodeBuffer = zeroBufferMatrice;
      _topology[nodeIndex+1].buffercounter = 0;
      _topology[nodeIndex+1].arVectorForPackets = zeroARVector;
    };
    //   while (flag) // until the decoder is full
    // {

    // -------------------------------------------------fake packet generation------------------------------------------------------------//
    std::vector<std::vector<uint8_t>> receivedPackets(_G, std::vector<uint8_t>(_packetSize, 0));
    // std::cout << "here";

    std::vector<uint8_t> coefficientVector = generateRandomVector(_G);
    // create an hpacket with the random data
    for (int j = 0; j < _G; j++)
    {
      receivedPackets[j] = generateRandomVector(_packetSize);
    };
    //----------------------------------------------------------------------------------------------------------------------------------------------//
    //----------------------------------------------------- key distribution -----------------------------------------------------------------------//
    // TO DO : This current version assign keypool to all nodes in the network
    // TO DO : Replace macnumber with keysetsize
    std::vector<std::vector<uint8_t>> key_pool(_keypoolSize, std::vector<uint8_t>(_packetSize + 1, 0));
    for (int i = 0; i < _keypoolSize; i++)
    {
      std::vector<uint8_t> newKey = generateRandomVector(_packetSize + 1);
      key_pool[i] = newKey;
    };

    _topology[0].keySet= key_pool;

    for (int k = 0; k < number_of_nonsource_nodes; k++)
    {
      // // Distribute keys
      _topology[k+1].keySet= keyDistributor(_keysetSize,key_pool);    // key distribution function willbe fixed
      //_topology[k + 1].keySet = key_pool;
    };
   
  
   
   // std::vector<std::vector<uint8_t>> assignedSet = keyDistributor(_keysetSize,key_pool);
   
    //       //-----------------------------------------------------------------------------------------------------------------------------------------------//

    std::vector<std::vector<uint8_t>> MACs;
    std::vector<uint8_t> appended_packet;
    int NumberOfLayers = 7;
    int Number_Of_Leaves = 2;
    //  std::cout << "here";
    int numVerticesInt = static_cast<int>(boost::num_vertices(_topology));
    std::vector<uint8_t> numOfIncomingPackets(std::vector<uint8_t>(numVerticesInt - 1, 0));

    std::vector<uint8_t> private_key = generateRandomVector(_keypoolSize + 1);
    hpacket p1(receivedPackets, MACs, key_pool, private_key, _keypoolSize, coefficientVector);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < _G; i++)
    { // i is packet index in a generation

      // select a random path from path list
      // Generate a random number between 0 and 4
      int pathIndex = 0;
      pathIndex = std::rand() % path_list.size(); // rand() % 5 will give a random number between 0 and 4


      //std::cout << "here";

      for (int j = 0; j < path_list[pathIndex].size(); j++) // one packet going through different nodes
      {
        if (_topology[path_list[pathIndex][j]].type == "Source")
        {
          //_topology[path_list[pathIndex][j]].input= zeroInputMatrice;
          _topology[path_list[pathIndex][j]].input = zeroInputVector;
          _topology[path_list[pathIndex][j]].input = receivedPackets[i];

          // Generate MACs and sign for a packet //
          p1.macCalculatorONEPACKET(_topology[path_list[pathIndex][j]].input, key_pool);
          p1.signCalculatorONEPACKET(_topology[path_list[pathIndex][j]].input, private_key);
          // Append generated MACs and sign to the end of packets

          // TO DO : Change appended packet to coded packet
          appended_packet = p1.packetAppenderONEPACKET(_topology[path_list[pathIndex][j]].input, p1.macCalculatorONEPACKET(_topology[path_list[pathIndex][j]].input, key_pool), p1.signCalculatorONEPACKET(receivedPackets[i], private_key));

          _topology[path_list[pathIndex][j]].output = zeroVector;
          _topology[path_list[pathIndex][j]].output = appended_packet;
          _topology[path_list[pathIndex][j]].SourceNodeOutput.push_back(appended_packet);

         // std::cout << "here";
          // Ask for sending to next node in the path
        }
        else if (_topology[path_list[pathIndex][j]].type == "Intermediate")
        {
        //  std::cout << "here";
          /* check the packet through verification,checkNumber++, */
          _topology[path_list[pathIndex][j]].input = zeroVector;
          _topology[path_list[pathIndex][j]].input = _topology[path_list[pathIndex][j - 1]].output;
          _topology[path_list[pathIndex][j]].path_index_vector = zeroARVector;
          // Assign AR value of the path that packets come from
          _topology[path_list[pathIndex][j]].arVectorForPackets.push_back(ARvector[pathIndex]);
          // Add input into the node buffer and increase buffer counter//
          _topology[path_list[pathIndex][j]].nonSourceNodeBuffer[_topology[path_list[pathIndex][j]].buffercounter] = _topology[path_list[pathIndex][j]].input;
          _topology[path_list[pathIndex][j]].buffercounter++;

         //  std::cout << "here";

          //  // Tree Ver and AR ver implementation //
          //  if (_topology[path_list[pathIndex][j]].buffercounter == _bufferSize)
          //  {
          //   std::vector<std::vector<std::vector<uint8_t>>> generated_tree=p1.treeGenerator(_topology[path_list[pathIndex][j]].nonSourceNodeBuffer, NumberOfLayers, Number_Of_Leaves,_packetSize);
          //   p1.treeVerifierNEW(generated_tree,NumberOfLayers,Number_Of_Leaves,_topology[path_list[pathIndex][j]].keySet,private_key);
          //   p1.arTreeVerifierNEW(generated_tree,_topology[path_list[pathIndex][j]].path_index_vector, _topology[path_list[pathIndex][j]].arVectorForPackets, NumberOfLayers, _topology[path_list[pathIndex][j]].keySet, private_key);
          //  }

          /////////////////////////////////////////////////////////////////

          bool MacResult = p1.macVerifier(_topology[path_list[pathIndex][j]].input, _topology[path_list[pathIndex][j]].keySet,key_pool);
        

          bool SignResult = p1.signVerifier(_topology[path_list[pathIndex][j]].input, private_key);
          // bool MacResult=  p1.macVerifier(_topology[path_list[pathIndex][j]].input,_topology[j].keySet);
          // bool SignResult= p1.signVerifier(_topology[path_list[pathIndex][j]].input,private_key);
          _topology[path_list[pathIndex][j]].checkNumber++;
          ////////  LOADING BUFFER FOR AR and TREE VERIFICATIONS ///////////////

          //////////////////////////////////////////////////////////////////////

          // verification shows healthy: add to healthyReceived counter & totalNodeSend++,
          if (MacResult && SignResult == true)
          {
            // TO DO : replace nonsourcenodeRLNV with recoding buffer
            _topology[path_list[pathIndex][j]].nonSourceNodeRLNC.push_back(_topology[path_list[pathIndex][j]].input);
            _topology[path_list[pathIndex][j]].healthyReceived++;
            // verification shows healthy but packet is not the same: falsePositiveEvents++
            if (_topology[path_list[pathIndex][0]].output != _topology[path_list[pathIndex][j]].input)
            { // Compare source output with verified intermediate node input
              _topology[path_list[pathIndex][j]].falsePositiveEvents++;
            };
            // TO DO : Replace output with RLNC packet
            _topology[path_list[pathIndex][j]].output= _topology[path_list[pathIndex][j]].input;
          }
          // verification shows polluted: pollutedReceived++ & PollutedDropped++
          else if ((MacResult == false) || (SignResult == false))
          {
            _topology[path_list[pathIndex][j]].pollutedReceived++;
            _topology[path_list[pathIndex][j]].pollutedDropped++;
            ARvector[pathIndex]++;
            break;
          };
          _topology[path_list[pathIndex][j]].totalNodeSend++;
        }
        else if (_topology[path_list[pathIndex][j]].type == "Adversary")
        {
      //     std::cout << "here";
          _topology[path_list[pathIndex][j]].input = zeroVector;
          _topology[path_list[pathIndex][j]].input = _topology[path_list[pathIndex][j - 1]].output;
         // std::cout << "here";
          // Pollute a packet with a probability
          double randomValue = dis(gen);
          _topology[path_list[pathIndex][j]].output = zeroVector;
          _topology[path_list[pathIndex][j]].output = _topology[path_list[pathIndex][j]].input;
          if (randomValue <= _topology[path_list[pathIndex][j]].attackProbability)
          {

            //_topology[path_list[pathIndex][j]].output = p1.pollutionGenerationONEPACKET(_topology[path_list[pathIndex][j]].input,_topology[path_list[pathIndex][j]].pollutedDropped);
            
            _topology[path_list[pathIndex][j]].output = p1.gf256_gaussian_elimination(_topology[path_list[pathIndex][j]].input,_topology[path_list[pathIndex][j]].keySet);

            _topology[path_list[pathIndex][j]].pollutedDropped++;
          //  std::cout << "here";
            ARvector[pathIndex]++;

          };

          _topology[path_list[pathIndex][j]].totalNodeSend++;
        }
        else if (_topology[path_list[pathIndex][j]].type == "Destination")
        {
      //    std::cout << "here";
          /* check the packet through verification,checkNumber++, */
          _topology[path_list[pathIndex][j]].input = zeroVector;
          _topology[path_list[pathIndex][j]].input = _topology[path_list[pathIndex][j - 1]].output;
          _topology[path_list[pathIndex][j]].path_index_vector[i] = pathIndex;
          _topology[path_list[pathIndex][j]].arVectorForPackets[i] = ARvector[pathIndex];
          // Add input into the node buffer and increase buffer counter//
          _topology[path_list[pathIndex][j]].nonSourceNodeBuffer[_topology[path_list[pathIndex][j]].buffercounter] = _topology[path_list[pathIndex][j]].input;
          _topology[path_list[pathIndex][j]].buffercounter++;



          if(_topology[path_list[pathIndex][j]].buffercounter == _bufferSize){
 
          
            std::vector<std::vector<std::vector<uint8_t>>> generated_tree = p1.treeGenerator(_topology[path_list[pathIndex][j]].nonSourceNodeBuffer, NumberOfLayers, Number_Of_Leaves, _packetSize);

            

            // AR verifier
            std::vector<int> ar_Results =  p1.arTreeVerifierNEW(generated_tree , _topology[path_list[pathIndex][j]].arVectorForPackets,NumberOfLayers,_topology[path_list[pathIndex][j]].keySet,key_pool,private_key);
            result_vector_arVerifier[generationIndex]= ar_Results;
            // Tree Verifier
             std::vector<int> tree_Results = p1.treeVerifierNEW(generated_tree, NumberOfLayers, Number_Of_Leaves,_topology[path_list[pathIndex][j]].keySet,key_pool, private_key);
             result_vector_treeVerifier[generationIndex]= tree_Results;
             // SImple Verifier
             std::vector<int> simple_Result = p1.simpleVerifierNEW(_topology[path_list[pathIndex][j]].nonSourceNodeBuffer, _topology[path_list[pathIndex][j]].keySet,key_pool, private_key);
             result_vector_simpleVerifier[generationIndex]= simple_Result;

         //    std::cout << "here";

          }

          //  // Tree Ver and AR ver implementation //
          //  if (_topology[path_list[pathIndex][j]].buffercounter == _bufferSize)
          //  {
          //   std::vector<std::vector<std::vector<uint8_t>>> generated_tree=p1.treeGenerator(_topology[path_list[pathIndex][j]].nonSourceNodeBuffer, NumberOfLayers, Number_Of_Leaves,_packetSize);
          //   std::vector<int> tree_ver_output = p1.treeVerifierNEW(generated_tree,NumberOfLayers,Number_Of_Leaves,_topology[path_list[pathIndex][j]].keySet,private_key);
          //   std::vector<int> ar_ver_output = p1.arTreeVerifierNEW(generated_tree,_topology[path_list[pathIndex][j]].path_index_vector, ARvector, NumberOfLayers, _topology[path_list[pathIndex][j]].keySet, private_key);
          //   _topology[path_list[pathIndex][j]].pollutedDropped= _topology[path_list[pathIndex][j]].pollutedDropped +  tree_ver_output[1] ;
          //  _topology[path_list[pathIndex][j]].pollutedDropped= _topology[path_list[pathIndex][j]].pollutedDropped +  ar_ver_output[1] ;
          //  }

          /////////////////////////////////////////////////////////////////

          //  Single packet verification //

          // bool MacResult=  p1.macVerifier(_topology[path_list[pathIndex][j]].input,_topology[j].keySet);
          // bool SignResult= p1.signVerifier(_topology[path_list[pathIndex][j]].input,private_key);



          // TO DO : UNCOMMENT LATER
          // bool MacResult = p1.macVerifier(_topology[path_list[pathIndex][j]].input, key_pool);
          // bool SignResult = p1.signVerifier(_topology[path_list[pathIndex][j]].input, private_key);
          // _topology[path_list[pathIndex][j]].checkNumber++;

          //////////////////////////////////////////////////////////////////////



          // verification shows healthy: add to healthyReceived counter & totalNodeSend++,

          //TO DO : UNCOMMENT LATER
          // if (MacResult && SignResult == true)
          // {
          //   _topology[path_list[pathIndex][j]].nonSourceNodeRLNC.push_back(_topology[path_list[pathIndex][j]].input);
          //   _topology[path_list[pathIndex][j]].healthyReceived++;
          //   // verification shows healthy but packet not the same: falsePositiveEvents++
          //   if (_topology[path_list[pathIndex][0]].output != _topology[path_list[pathIndex][j]].input)
          //   { // Compare source output with verified intermediate node input
          //     _topology[path_list[pathIndex][j]].falsePositiveEvents++;
          //   };
          // }
          // // verification shows polluted: pollutedReceived++ & PollutedDropped++
          // else if (MacResult || SignResult == false)
          // {
          //   _topology[path_list[pathIndex][j]].pollutedReceived++;
          //   _topology[path_list[pathIndex][j]].pollutedDropped++;
          //   ARvector[pathIndex]++;
          // };


          // };
        };
      };
    };

if(generationIndex==_GG-1){
   std::cout << "here";}
    

    
  };
};

// void simulation(Graph _topology, std::vector<std::vector<Graph::vertex_descriptor>> path_list, int _GG, int _G, int _fieldSize, int _packetSize, int _keypoolSize, int _keysetSize, int Number_of_Pollution,int _bufferSize)

//  for (int generationIndex = 0; generationIndex < _GG; generationIndex++)
//   {
//     bool flag = true;
//     // Initialize buffers of each nonsource nodes in each generation
//     for (int nodeIndex = 0; nodeIndex < number_of_nonsource_nodes; nodeIndex++)
//     {
//       _topology[nodeIndex].nonSourceNodeBuffer= zeroBufferMatrice ;
//       _topology[nodeIndex].buffercounter=0;
//       _topology[nodeIndex].arVectorForPackets= zeroARVector;
//     }

//     while (flag) // until the decoder is full
//     {
//       // select a random path from path list
//       std::srand(std::time(nullptr));
//       // Generate a random number between 0 and 4
//       int pathIndex = std::rand() % path_list.size(); // rand() % 5 will give a random number between 0 and 4

//       // -------------------------------------------------fake packet generation------------------------------------------------------------//
//       std::vector<std::vector<uint8_t>> receivedPackets(_G, std::vector<uint8_t>(_packetSize, 0));
//       // std::cout << "here";

//       std::vector<uint8_t> coefficientVector = generateRandomVector(_G);
//       // create an hpacket with the random data
//       for (int j = 0; j < _G; j++)
//       {

//         std::vector<uint8_t> cs1 = generateRandomVector(_packetSize);
//         receivedPackets[j] = cs1;
//       };

//       //----------------------------------------------------------------------------------------------------------------------------------------------//
//       //----------------------------------------------------- key distribution -----------------------------------------------------------------------//
//         int MACNumber=5;
//       std::vector<std::vector<uint8_t>> key1(MACNumber, std::vector<uint8_t>(_packetSize + 1, 0));
//     for (int i = 0; i < MACNumber; i++)
//     {
//       std::vector<uint8_t> newKey = generateRandomVector(_packetSize + 1);
//       key1[i] = newKey;
//     };

//       for(int k=0; k<number_of_nonsource_nodes-1;k++){
//       // Generate a random number between 0 and
//             _topology[k+1].keySet= keyDistributor(_keysetSize,key1);
//       };
//       //-----------------------------------------------------------------------------------------------------------------------------------------------//
//       std::vector<std::vector<uint8_t>> MACs;
//       std::vector<uint8_t> appended_packet;
//       int NumberOfLayers =4;
//       int Number_Of_Leaves = 2;
//      std::cout << "here";
//       int numVerticesInt = static_cast<int>(boost::num_vertices(_topology));
//       std::vector<uint8_t>numOfIncomingPackets(std::vector<uint8_t>(numVerticesInt-1,0));

//       std::vector<uint8_t> private_key = generateRandomVector(MACNumber + 1);
//       hpacket p1(receivedPackets, MACs, key1, private_key, MACNumber, coefficientVector);
//       std::random_device rd;
//       std::mt19937 gen(rd());
//       std::uniform_real_distribution<double> dis(0.0, 1.0);

//     for(int i=0; i<_G; i++){   // i is packet index in a generation
//          std::cout << "here" << i << std::endl;

//       for (int j = 0; j < path_list[pathIndex].size(); j++) // one packet going through different nodes
//       {
//       if(_topology[path_list[pathIndex][j]].type == "Source"){
//         //_topology[path_list[pathIndex][j]].input= zeroInputMatrice;
//         _topology[path_list[pathIndex][j]].input= zeroInputVector;
//         _topology[path_list[pathIndex][j]].input= receivedPackets[i];

//         // Generate MACs and sign for a packet //
//         p1.macCalculatorONEPACKET(_topology[path_list[pathIndex][j]].input,key1);
//         p1.signCalculatorONEPACKET(_topology[path_list[pathIndex][j]].input,private_key);
//         // Append generated MACs and sign to the end of packets
//         appended_packet = p1.packetAppenderONEPACKET(_topology[path_list[pathIndex][j]].input,p1.macCalculatorONEPACKET(_topology[path_list[pathIndex][j]].input,key1),p1.signCalculatorONEPACKET(receivedPackets[i],private_key));

//         _topology[path_list[pathIndex][j]].output=zeroVector;
//         _topology[path_list[pathIndex][j]].output=appended_packet;
//        //Ask for sending to next node in the path

//       }
//         else if (_topology[path_list[pathIndex][j]].type == "Intermediate")
//         {
//           /* check the packet through verification,checkNumber++, */
//           _topology[path_list[pathIndex][j]].input= zeroVector;
//           _topology[path_list[pathIndex][j]].input= _topology[path_list[pathIndex][j-1]].output;
//           _topology[path_list[pathIndex][j]].path_index_vector= zeroARVector;
//            // Assign AR value of the path that packets come from
//            _topology[path_list[pathIndex][j]].arVectorForPackets.push_back(ARvector[pathIndex]);
//           // Add input into the node buffer and increase buffer counter//
//          _topology[path_list[pathIndex][j]].nonSourceNodeBuffer.push_back(_topology[path_list[pathIndex][j]].input);
//          _topology[path_list[pathIndex][j]].buffercounter++;

//          // Tree Ver and AR ver implementation //
//          if (_topology[path_list[pathIndex][j]].buffercounter == _bufferSize)
//          {
//           std::vector<std::vector<std::vector<uint8_t>>> generated_tree=p1.treeGenerator(_topology[path_list[pathIndex][j]].nonSourceNodeBuffer, NumberOfLayers, Number_Of_Leaves,_packetSize);
//           p1.treeVerifierNEW(generated_tree,NumberOfLayers,Number_Of_Leaves,_topology[path_list[pathIndex][j]].keySet,private_key);
//           p1.arTreeVerifierNEW(generated_tree,_topology[path_list[pathIndex][j]].path_index_vector, _topology[path_list[pathIndex][j]].arVectorForPackets, NumberOfLayers, _topology[path_list[pathIndex][j]].keySet, private_key);
//          }

//          /////////////////////////////////////////////////////////////////

//           bool MacResult=  p1.macVerifier(_topology[path_list[pathIndex][j]].input,_topology[j].keySet);
//           bool SignResult= p1.signVerifier(_topology[path_list[pathIndex][j]].input,private_key);
//           _topology[path_list[pathIndex][j]].checkNumber++;
//           ////////  LOADING BUFFER FOR AR and TREE VERIFICATIONS ///////////////

//           //////////////////////////////////////////////////////////////////////

//           // verification shows healthy: add to healthyReceived counter & totalNodeSend++,
//           if(MacResult && SignResult == true){
//             _topology[path_list[pathIndex][j]].nonSourceNodeRLNC.push_back(_topology[path_list[pathIndex][j]].input);
//             _topology[path_list[pathIndex][j]].healthyReceived++;
//           // verification shows healthy but packet not the same: falsePositiveEvents++
//             if(_topology[path_list[pathIndex][0]].output !=  _topology[path_list[pathIndex][j]].input){   //Compare source output with verified intermediate node input
//                _topology[path_list[pathIndex][j]].falsePositiveEvents++;
//             };
//           }
//           // verification shows polluted: pollutedReceived++ & PollutedDropped++
//              else if(MacResult || SignResult == false){
//               _topology[path_list[pathIndex][j]].pollutedReceived++;
//               _topology[path_list[pathIndex][j]].pollutedDropped++;
//               ARvector[pathIndex]++;
//           };

//         }
//         else if (_topology[path_list[pathIndex][j]].type == "Adversary")
//         {
//           _topology[path_list[pathIndex][j]].input= zeroVector;
//           _topology[path_list[pathIndex][j]].input= _topology[path_list[pathIndex][j-1]].output;

//         // Pollute a packet with a probability
//           double randomValue = dis(gen);
//           _topology[path_list[pathIndex][j]].output=zeroVector;
//           if(randomValue <= _topology[path_list[pathIndex][j]].attackProbability){

//             _topology[path_list[pathIndex][j]].output=p1.pollutionGenerationONEPACKET(_topology[path_list[pathIndex][j]].input);
//           };

//           _topology[path_list[pathIndex][j]].output=_topology[path_list[pathIndex][j]].input;
//           _topology[path_list[pathIndex][j]].totalNodeSend++;

//         }
//         else if (_topology[path_list[pathIndex][j]].type == "Destination")
//         {
// /* check the packet through verification,checkNumber++, */
//           _topology[path_list[pathIndex][j]].input= zeroVector;
//           _topology[path_list[pathIndex][j]].input= _topology[path_list[pathIndex][j-1]].output;
//           _topology[path_list[pathIndex][j]].path_index_vector=zeroARVector;
//           // Add input into the node buffer and increase buffer counter//
//          _topology[path_list[pathIndex][j]].nonSourceNodeBuffer.push_back(_topology[path_list[pathIndex][j]].input);
//          _topology[path_list[pathIndex][j]].buffercounter++;

//          // Tree Ver and AR ver implementation //
//          if (_topology[path_list[pathIndex][j]].buffercounter == _bufferSize)
//          {
//           std::vector<std::vector<std::vector<uint8_t>>> generated_tree=p1.treeGenerator(_topology[path_list[pathIndex][j]].nonSourceNodeBuffer, NumberOfLayers, Number_Of_Leaves,_packetSize);
//           std::vector<int> tree_ver_output = p1.treeVerifierNEW(generated_tree,NumberOfLayers,Number_Of_Leaves,_topology[path_list[pathIndex][j]].keySet,private_key);
//           std::vector<int> ar_ver_output = p1.arTreeVerifierNEW(generated_tree,_topology[path_list[pathIndex][j]].path_index_vector, ARvector, NumberOfLayers, _topology[path_list[pathIndex][j]].keySet, private_key);
//           _topology[path_list[pathIndex][j]].pollutedDropped= _topology[path_list[pathIndex][j]].pollutedDropped +  tree_ver_output[1] ;
//          _topology[path_list[pathIndex][j]].pollutedDropped= _topology[path_list[pathIndex][j]].pollutedDropped +  ar_ver_output[1] ;
//          }

//          /////////////////////////////////////////////////////////////////

//          //  Single packet verification //

//           bool MacResult=  p1.macVerifier(_topology[path_list[pathIndex][j]].input,_topology[j].keySet);
//           bool SignResult= p1.signVerifier(_topology[path_list[pathIndex][j]].input,private_key);
//           _topology[path_list[pathIndex][j]].checkNumber++;

//           //////////////////////////////////////////////////////////////////////

//           // verification shows healthy: add to healthyReceived counter & totalNodeSend++,
//           if(MacResult && SignResult == true){
//             _topology[path_list[pathIndex][j]].nonSourceNodeRLNC.push_back(_topology[path_list[pathIndex][j]].input);
//             _topology[path_list[pathIndex][j]].healthyReceived++;
//           // verification shows healthy but packet not the same: falsePositiveEvents++
//             if(_topology[path_list[pathIndex][0]].output !=  _topology[path_list[pathIndex][j]].input){   //Compare source output with verified intermediate node input
//                _topology[path_list[pathIndex][j]].falsePositiveEvents++;
//             };
//           }
//           // verification shows polluted: pollutedReceived++ & PollutedDropped++
//              else if(MacResult || SignResult == false){
//               _topology[path_list[pathIndex][j]].pollutedReceived++;
//               _topology[path_list[pathIndex][j]].pollutedDropped++;
//                ARvector[pathIndex]++;
//           };

//           // pass all the coefficients to a vector to identify the rank
//           // if the rank is equal to G, then break from while true
//           bool flag = false;
//         }
//       };
//     };
//   };
//   };