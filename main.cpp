#include <iostream>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <random>
//#include "headers/rlnc_encoder.h"
#include "headers/ff.h"
#include "headers/pff.h"
 #include "headers/cFunctions.h"
//#include "headers/packet.h"
//#include "headers/rlnc_decoder.h"
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

//  ff fff(256);
// pff fff(127);

std::vector<uint8_t> generateRandomVector(int size, int _fieldsize)

{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<uint8_t> randomVector(size);

  if (_fieldsize == 256)
  {
    std::uniform_int_distribution<uint8_t> dis(1, _fieldsize - 1);
    for (int i = 0; i < size; ++i)
    {
      randomVector[i] = dis(gen);
    }
  }
  else
  {

    std::uniform_int_distribution<uint8_t> dis(1, _fieldsize - 1);
    for (int i = 0; i < size; ++i)
    {
      randomVector[i] = dis(gen);
    };
  };

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
std::vector<int> chosenNumbers(int _KeysetSize, int _keypoolSize)
{
  // std::cout << "here";

  std::vector<int> chosenSet;

  std::vector<int> chosenNumbers(_KeysetSize, 0);
  // Create a discrete distribution based on the probabilities
  std::uniform_int_distribution<int> distribution(0, _keypoolSize - 1);
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

  return chosenNumbers;
};
//
std::vector<std::vector<uint8_t>> keyDistributor(std::vector<std::vector<uint8_t>> KeyPool, std::vector<int> keyIndexes)
{
  // std::cout << "here";

  std::vector<std::vector<uint8_t>> assignedKeyset(keyIndexes.size(), std::vector<uint8_t>(KeyPool[0].size(), 0));

  //  std::vector<int> chosenNumbers(keyIndexes.size(), 0);
  // Create a discrete distribution based on the probabilities

  for (int i = 0; i < keyIndexes.size(); i++)
  {

    assignedKeyset[i] = KeyPool[keyIndexes[i]];
  };
  return assignedKeyset;
};

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
  double attackProbability;
  std::vector<uint8_t> output;
  std::vector<uint8_t> input;
  std::vector<int> arVectorForPackets;
  std::vector<int> path_index_vector;
  std::vector<int> key_index_vector;
  std::vector<int> sorted_key_index_vector;
};
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexProperties> Graph;

void simulation(Graph _topology, std::vector<std::vector<Graph::vertex_descriptor>> path_list, int _GG, int _G, int _fieldSize, int _packetSize, int _keypoolSize, int _keysetSize, int Number_of_Pollution, int _bufferSize, double attackProbability);

// ........................................................
int main(int argc, char *argv[])
{
  int _input_generation = 0;
  int _input_fieldSize = 0;
  int _input_symbolSize = 0;

  int opt;

  // Display the number of command-line arguments
  std::cout << "Number of arguments: " << argc << std::endl;

  // Updated getopt string to include 'g' and 'f'
  while ((opt = getopt(argc, argv, "m:n:f:")) != -1)
  {
    switch (opt)

    {
    case 'm':
      _input_generation = std::stoi(optarg);
      break;
    case 'n':
      _input_symbolSize = std::stoi(optarg);
      break;
    case 'f':
      _input_fieldSize = std::stoi(optarg);
      break;
    default:
      std::cerr << "Usage: " << argv[0] << " -g <generation> -s <fieldSize> -f <symbolSize>" << std::endl;
    }
  }

  // Output the parsed values
  std::cout << "_input_generation: " << _input_generation << std::endl;
  std::cout << "_input_fieldSize: " << _input_fieldSize << std::endl;
  std::cout << "_input_symbolSize: " << _input_symbolSize << std::endl;

  

  // generate topology
  Graph g;

  // Add vertices to the graph and assign properties
  auto v0 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Source Node
  auto v1 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Intermediate_1 Node
  auto v2 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Compromised Node
  auto v3 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Intermediate_2 Node
  auto v4 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Destination Node

  // Create a connetion between Nodes
  boost::add_edge(v0, v1, g); // From Source to Intermediate_1
  boost::add_edge(v0, v3, g); // From Source to Intermediate_2
  boost::add_edge(v0, v2, g); // From Source to Adversary
  boost::add_edge(v1, v2, g); // From Intermediate_1 to Adversary
  boost::add_edge(v3, v2, g); // From Adversary to Intermediate
  boost::add_edge(v1, v3, g); // From Intermediate_1 to Intermediate_2
  boost::add_edge(v3, v4, g); // From  Intermediate_2 to Destination
  boost::add_edge(v1, v4, g); // From  Intermediate_1 to Destination
  boost::add_edge(v2, v4, g); // From  Adversary to Destination

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
  int _G = _input_generation; // generation size (m)
  int _fieldSize = _input_fieldSize; // finite field size (f)
  int _symbolSize = _input_symbolSize; // (n)
  int _packetSize = _symbolSize+ _G; // m+n
  int _keypoolSize = 42;
  int _keysetSize = 4;
  int Number_of_Pollution = 1;

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
  // int _G = 64; // generationsize
//  int min_G = 8;
  // int max_G = 128;
  double min_att_rate = 0.4;
  double max_att_rate = 0.405;
  for (double attackProbability = min_att_rate; attackProbability < max_att_rate; attackProbability)
  {
  //  for (int _G = min_G; _G < (max_G + 1); _G)
    //{
      int _bufferSize = _G;
      g[v4].path_index_vector = std::vector<int>(_bufferSize, 0);
      g[v4].arVectorForPackets = std::vector<int>(_bufferSize, 0);
      simulation(g, path_list, _GG, _G, _fieldSize, _symbolSize, _packetSize, _keypoolSize, _keysetSize, Number_of_Pollution, _bufferSize, attackProbability); //
     // _G += _G;
    //}
    attackProbability += 0.2;
  }
}

void simulation(Graph _topology, std::vector<std::vector<Graph::vertex_descriptor>> path_list, int _GG, int _G, int _fieldSize, int _symbolSize, int _packetSize, int _keypoolSize, int _keysetSize, int Number_of_Pollution, int _bufferSize, double attackProbability)
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

  std::vector<std::vector<int>> result_vector_treeVerifier(_GG, std::vector<int>(2, 0));

  std::vector<std::vector<int>> result_vector_arVerifier(_GG, std::vector<int>(2, 0));

  std::vector<std::vector<int>> result_vector_simpleVerifier(_GG, std::vector<int>(2, 0));

  std::vector<double> polluted_packets(_GG, 0);

  std::vector<std::vector<int>> key_difference_matrice(_GG, std::vector<int>(_keysetSize + 1, 0));

  std::string filename = "./Results/BufferSize" + std::to_string(256) + "-" + std::to_string(512);
  filename += "Field Size" + std::to_string(_fieldSize);
  filename += "Generation Number" + std::to_string(_GG);
  filename += "MAC Size" + std::to_string(_keypoolSize);
  filename += "Keyset Size" + std::to_string(_keysetSize);
  filename += "PacketSize:" + std::to_string(_packetSize);
  filename += ".txt";

  std::ofstream outputFile(filename, std::ios::app);
  /////////////////-----------------------------------------------------------------------------------------------------//////////
  std::string filename_key_dist = "./Results/BufferSize" + std::to_string(256) + "-" + std::to_string(512);
  filename_key_dist += "MAC Size" + std::to_string(_keypoolSize);
  filename_key_dist += "Generation Number" + std::to_string(_GG);
  filename_key_dist += "Keyset Size" + std::to_string(_keysetSize);
  filename_key_dist += ".txt";

  std::ofstream outputFile_key_dist(filename_key_dist, std::ios::app);

  /////////////////-----------------------------------------------------------------------------------------------------------/////////////////////////////////////////////////////
  // std::string filename_all_data_simple = "./Results/BufferSize" + std::to_string(8) + "-" + std::to_string(128);
  // filename_all_data_simple += "Simple Verifier";
  // filename_all_data_simple += "Field Size" + std::to_string(_fieldSize);
  // filename_all_data_simple += "Generation Number" + std::to_string(_GG);
  // filename_all_data_simple += "MAC" + std::to_string(_keypoolSize);
  // filename_all_data_simple += "Keyset Size" + std::to_string(_keysetSize);
  // filename_all_data_simple += "PacketSize:" + std::to_string(_packetSize);
  // filename_all_data_simple += ".txt";

  // std::ofstream outputFile_all_data_simple(filename_all_data_simple, std::ios::app);
  /////////////////-----------------------------------------------------------------------------------------------------------/////////////////////////////////////////////////////
  std::string filename_all_data_tree = "./Results/BufferSize" + std::to_string(256) + "-" + std::to_string(512);
  filename_all_data_tree += "Tree Verifier";
  filename_all_data_tree += "Field Size" + std::to_string(_fieldSize);
  filename_all_data_tree += "Generation Number" + std::to_string(_GG);
  filename_all_data_tree += "MAC" + std::to_string(_keypoolSize);
  filename_all_data_tree += "Keyset Size" + std::to_string(_keysetSize);
  filename_all_data_tree += "PacketSize:" + std::to_string(_packetSize);
  filename_all_data_tree += ".txt";

  std::ofstream outputFile_all_data_tree(filename_all_data_tree, std::ios::app);
  /////////////////-----------------------------------------------------------------------------------------------------------/////////////////////////////////////////////////////
  std::string filename_all_data_ar = "./Results/BufferSize" + std::to_string(256) + "-" + std::to_string(512);
  filename_all_data_ar += "AR Verifier";
  filename_all_data_ar += "Field Size" + std::to_string(_fieldSize);
  filename_all_data_ar += "Generation Number" + std::to_string(_GG);
  filename_all_data_ar += "MAC" + std::to_string(_keypoolSize);
  filename_all_data_ar += "Keyset Size" + std::to_string(_keysetSize);
  filename_all_data_ar += "PacketSize:" + std::to_string(_packetSize);
  filename_all_data_ar += ".txt";

  std::ofstream outputFile_all_data_ar(filename_all_data_ar, std::ios::app);

  std::srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());

  //----------------------------------------------  Start Each Generation  -------------------------------------------------------//
  for (int generationIndex = 0; generationIndex < _GG; generationIndex++)
  {
    bool flag = true;
    // Initialize buffers of each nonsource nodes in each generation
    for (int nodeIndex = 0; nodeIndex < number_of_nonsource_nodes; nodeIndex++)
    {
      _topology[nodeIndex + 1].nonSourceNodeBuffer = zeroBufferMatrice;
      _topology[nodeIndex + 1].buffercounter = 0;
      _topology[nodeIndex + 1].arVectorForPackets = zeroARVector;
    };
    //   while (flag) // until the decoder is full
    // {
    _topology[2].pollutedDropped = 0;

    // -------------------------------------------------packet & coefficient generation------------------------------------------------------------//
    std::vector<std::vector<uint8_t>> receivedPackets(_G, std::vector<uint8_t>(_symbolSize, 0));
    // std::cout << "here";
    std::vector<std::vector<uint8_t>> coefficientMatrice(_G, std::vector<uint8_t>(_G, 0));
    // create an hpacket with the random data
    for (int j = 0; j < _G; j++)
    {
      receivedPackets[j] = generateRandomVector(_symbolSize, _fieldSize);  // with size n
      coefficientMatrice[j] = generateRandomVector(_G, _fieldSize);        // with size m
    };
    //----------------------------------------------------------------------------------------------------------------------------------------------//
    //----------------------------------------------------- key distribution -----------------------------------------------------------------------//
    // TO DO : This current version assign keypool to all nodes in the network
    // TO DO : Replace macnumber with keysetsize
    std::vector<std::vector<uint8_t>> key_pool(_keypoolSize, std::vector<uint8_t>(_packetSize + 1, 0));
    for (int i = 0; i < _keypoolSize; i++)
    {
      std::vector<uint8_t> newKey = generateRandomVector((_packetSize + 1), _fieldSize);
      key_pool[i] = newKey;
    };

    _topology[0].keySet = key_pool;

    for (int k = 0; k < number_of_nonsource_nodes; k++)
    {
      // // Distribute keys
      _topology[k + 1].key_index_vector = chosenNumbers(_keysetSize, _keypoolSize);
      _topology[k + 1].keySet = keyDistributor(key_pool, _topology[k + 1].key_index_vector);

      //_topology[k + 1].keySet = key_pool;
    };
    _topology[4].key_index_vector = chosenNumbers(_keysetSize, _keypoolSize);
    _topology[4].keySet = keyDistributor(key_pool, _topology[4].key_index_vector);

    // std::vector<std::vector<uint8_t>> assignedSet = keyDistributor(_keysetSize,key_pool);

    //       //-----------------------------------------------------------------------------------------------------------------------------------------------//

    std::vector<std::vector<uint8_t>> MACs;
    std::vector<uint8_t> appended_packet;
    int NumberOfLayers = std::log2(_G) + 1;
    int Number_Of_Leaves = 2;
    //  std::cout << "here";
    int numVerticesInt = static_cast<int>(boost::num_vertices(_topology));
    std::vector<uint8_t> numOfIncomingPackets(std::vector<uint8_t>(numVerticesInt - 1, 0));

    std::vector<uint8_t> private_key = generateRandomVector((_keypoolSize + 1), _fieldSize);
    hpacket p1(receivedPackets, MACs, key_pool, private_key, _keypoolSize, coefficientMatrice);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    //   int fp_counter=0;
    //   for(int it=0; it<10000000; it++){
    //   std::vector<std::vector<uint8_t>> _test_packets(5,std::vector<uint8_t>(_packetSize,0));
    //   for(int x=0; x<5; x++){
    //      _test_packets[x]= generateRandomVector(_packetSize,_fieldSize);
    //   };
    //  bool sum_Result= p1.fp_checker(_packetSize,_fieldSize, 5, _test_packets);
    //  if(sum_Result==true){
    //   fp_counter++;
    //  };
    // };

    for (int i = 0; i < _G; i++)
    { // i is packet index in a generation

      // select a random path from path list
      // Generate a random number between 0 and 4
      int pathIndex = 0;
      pathIndex = std::rand() % path_list.size(); // rand() % 5 will give a random number between 0 and 4

      // do{
      //   _topology[2].key_index_vector = chosenNumbers(_keysetSize,_keypoolSize);
      //   _topology[2].keySet = keyDistributor(key_pool,_topology[2].key_index_vector);
      //   std::sort(_topology[4].key_index_vector.begin(), _topology[4].key_index_vector.end());
      //   std::sort(_topology[2].key_index_vector.begin(), _topology[2].key_index_vector.end());
      // }      while( _topology[2].key_index_vector == _topology[4].key_index_vector);
      int same_key_counter = 0;
      _topology[2].key_index_vector = chosenNumbers(_keysetSize, _keypoolSize);
      _topology[2].keySet = keyDistributor(key_pool, _topology[2].key_index_vector);
      std::sort(_topology[4].key_index_vector.begin(), _topology[4].key_index_vector.end());
      std::sort(_topology[2].key_index_vector.begin(), _topology[2].key_index_vector.end());
      for (int ii = 0; ii < _keysetSize; ii++)
      {
        for (int jj = 0; jj < _keysetSize; jj++)
        {

          if (_topology[2].key_index_vector[ii] == _topology[4].key_index_vector[jj])
          {
            same_key_counter++;
          }
        }
      }
      key_difference_matrice[generationIndex][same_key_counter]++;

      // }      while( _topology[2].key_index_vector == _topology[4].key_index_vector);

      // std::cout << "here";
std::vector<std::vector<uint8_t>> transmitted_packets;

transmitted_packets  = p1.hmac_encoder(coefficientMatrice, receivedPackets);   // encoding operations


      for (int j = 0; j < path_list[pathIndex].size(); j++) // one packet going through different nodes
      {
        if (_topology[path_list[pathIndex][j]].type == "Source")
        {
          //_topology[path_list[pathIndex][j]].input= zeroInputMatrice;
          _topology[path_list[pathIndex][j]].input = zeroInputVector;
          _topology[path_list[pathIndex][j]].input = transmitted_packets[i];

          // Generate MACs and sign for a packet //
          // p1.macCalculatorONEPACKET(_topology[path_list[pathIndex][j]].input, key_pool);
          // p1.signCalculatorONEPACKET(_topology[path_list[pathIndex][j]].input, private_key);
          // Append generated MACs and sign to the end of packets

          // TO DO : Change appended packet to coded packet
          appended_packet = p1.packetAppenderONEPACKET(_topology[path_list[pathIndex][j]].input, p1.macCalculatorONEPACKET(_topology[path_list[pathIndex][j]].input, key_pool), p1.signCalculatorONEPACKET(_topology[path_list[pathIndex][j]].input, private_key));

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

          bool MacResult = p1.macVerifier(_topology[path_list[pathIndex][j]].input, _topology[path_list[pathIndex][j]].keySet, key_pool);

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
            _topology[path_list[pathIndex][j]].output = _topology[path_list[pathIndex][j]].input;
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
          _topology[path_list[pathIndex][j]].attackProbability = attackProbability;
          if (randomValue <= _topology[path_list[pathIndex][j]].attackProbability)
          {

            // _topology[path_list[pathIndex][j]].output = p1.pollutionGenerationONEPACKET(_topology[path_list[pathIndex][j]].input,_topology[path_list[pathIndex][j]].pollutedDropped);

            _topology[path_list[pathIndex][j]].output = p1.rref(_topology[path_list[pathIndex][j]].keySet, key_pool, _topology[path_list[pathIndex][j]].input);

            //  std::vector<uint8_t> pol_test = p1.rref(_topology[path_list[pathIndex][j]].keySet,key_pool,_topology[path_list[pathIndex][j]].input);

            // bool MacResult_test = p1.macVerifier(_topology[path_list[pathIndex][j]].input, _topology[path_list[pathIndex][j]].keySet,key_pool);

            // MacResult_test = p1.macVerifier(pol_test, _topology[path_list[pathIndex][j]].keySet,key_pool);

            // MacResult_test = p1.macVerifier(pol_test, key_pool,key_pool);

            // bool SignResult_test = p1.signVerifier(_topology[path_list[pathIndex][j]].input, private_key);
            //_topology[path_list[pathIndex][j]].output = p1.gf256_gaussian_elimination(_topology[path_list[pathIndex][j]].input,_topology[path_list[pathIndex][j]].keySet);

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

          // _topology[path_list[pathIndex][j]].keySet=key_pool;

          if (_topology[path_list[pathIndex][j]].buffercounter == _bufferSize)
          {

            std::vector<std::vector<std::vector<uint8_t>>> generated_tree = p1.treeGenerator(_topology[path_list[pathIndex][j]].nonSourceNodeBuffer, NumberOfLayers, Number_Of_Leaves, _packetSize);

            // AR verifier
            std::vector<int> ar_Results = p1.arTreeVerifierNEW(generated_tree, _topology[path_list[pathIndex][j]].arVectorForPackets, NumberOfLayers, _topology[path_list[pathIndex][j]].keySet, key_pool, private_key);
            result_vector_arVerifier[generationIndex] = ar_Results;

            outputFile_all_data_ar << "PacketSize:" << _packetSize << "-" << "BufferSize:" << _G << "-" << "Attack Probability:" << attackProbability
                                   << "-" << "False Positivity AR:" << _topology[2].pollutedDropped << "-" << "Detected Pollution Number" << ar_Results[1] << "-" << "AR Ver Check Number:" << ar_Results[0] << std::endl;
            outputFile_all_data_ar.flush();

            //   std::cout << "here";
            // Tree Verifier
            std::vector<int> tree_Results = p1.treeVerifierNEW(generated_tree, NumberOfLayers, Number_Of_Leaves, _topology[path_list[pathIndex][j]].keySet, key_pool, private_key);
            result_vector_treeVerifier[generationIndex] = tree_Results;

            outputFile_all_data_tree << "PacketSize:" << _packetSize << "-" << "BufferSize:" << _G << "-" << "Attack Probability:" << attackProbability
                                     << "-" << "False Positivity Tree:" << _topology[2].pollutedDropped << "-" << "Detected Pollution Number" << tree_Results[1] << "-" << "Batch Ver Check Number:" << tree_Results[0] << std::endl;
            outputFile_all_data_tree.flush();

            // SImple Verifier
            std::vector<int> simple_Result = p1.simpleVerifierNEW(_topology[path_list[pathIndex][j]].nonSourceNodeBuffer, _topology[path_list[pathIndex][j]].keySet, key_pool, private_key);
            result_vector_simpleVerifier[generationIndex] = simple_Result;

            //  outputFile_all_data_simple << "PacketSize:" << _packetSize<< "-"<< "BufferSize:" << _G<< "-"<< "Attack Probability:"<<  attackProbability
            //     << "-"<< "False Positivity Simple:" <<  _topology[2].pollutedDropped << "-"<< "Detected Pollution Number" <<  simple_Result[1] << "-"<< "Simple Ver Check Number:" <<  simple_Result[0] << std::endl;
            // outputFile_all_data_simple.flush();

            // std::cout << "here";
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

          // TO DO : UNCOMMENT LATER
          //  if (MacResult && SignResult == true)
          //  {
          //    _topology[path_list[pathIndex][j]].nonSourceNodeRLNC.push_back(_topology[path_list[pathIndex][j]].input);
          //    _topology[path_list[pathIndex][j]].healthyReceived++;
          //    // verification shows healthy but packet not the same: falsePositiveEvents++
          //    if (_topology[path_list[pathIndex][0]].output != _topology[path_list[pathIndex][j]].input)
          //    { // Compare source output with verified intermediate node input
          //      _topology[path_list[pathIndex][j]].falsePositiveEvents++;
          //    };
          //  }
          //  // verification shows polluted: pollutedReceived++ & PollutedDropped++
          //  else if (MacResult || SignResult == false)
          //  {
          //    _topology[path_list[pathIndex][j]].pollutedReceived++;
          //    _topology[path_list[pathIndex][j]].pollutedDropped++;
          //    ARvector[pathIndex]++;
          //  };

          // };
        };
      };
    };

    // if(generationIndex==_GG-1){
    //    std::cout << "here";}
    polluted_packets[generationIndex] = _topology[2].pollutedDropped;

  }; // for loop for each generation
  // ----------------------- AVERAGE VALUE CALCULATION STARTS -------------------------------------------//
  int pol_sum_simple = 0;
  int pol_sum_tree = 0;
  int pol_sum_ar = 0;
  int sum_simple = 0;
  int sum_tree = 0;
  int sum_ar = 0;
  double pol = 0;
  int counter = 0;
  std::vector<int> sum_key_diff_matrice(_keysetSize + 1, 0);

  for (int ii = 0; ii < (_keysetSize + 1); ii++)
  {
    for (int jj = 0; jj < key_difference_matrice.size(); jj++)
    {
      sum_key_diff_matrice[ii] += key_difference_matrice[jj][ii];
    }
  }

  for (int i = 0; i < result_vector_simpleVerifier.size(); i++)
  {
    if ((polluted_packets[i] < result_vector_simpleVerifier[i][1]) || (result_vector_simpleVerifier[i][0] == 0))
    {
      polluted_packets[i] = 0;
      result_vector_simpleVerifier[i] = {0, 0};
      result_vector_treeVerifier[i] = {0, 0};
      result_vector_arVerifier[i] = {0, 0};
      counter++;
    }
    pol_sum_simple += result_vector_simpleVerifier[i][1];
    sum_simple += result_vector_simpleVerifier[i][0];
    pol_sum_tree += result_vector_treeVerifier[i][1];
    sum_tree += result_vector_treeVerifier[i][0];
    pol_sum_ar += result_vector_arVerifier[i][1];
    sum_ar += result_vector_arVerifier[i][0];
    pol += polluted_packets[i];
  }
  double average_simple = static_cast<double>(sum_simple) / (result_vector_simpleVerifier.size() - counter);
  double pol_average_simple = static_cast<double>(pol_sum_simple) / (result_vector_simpleVerifier.size() - counter);
  double average_tree = static_cast<double>(sum_tree) / (result_vector_simpleVerifier.size() - counter);
  double pol_average_tree = static_cast<double>(pol_sum_tree) / (result_vector_simpleVerifier.size() - counter);
  double average_ar = static_cast<double>(sum_ar) / (result_vector_simpleVerifier.size() - counter);
  double pol_average_ar = static_cast<double>(pol_sum_ar) / (result_vector_simpleVerifier.size() - counter);
  double fp_ar = 100 * (pol - pol_sum_ar) / pol;
  double fp_simple = 100 * (pol - pol_sum_simple) / pol;
  double fp_tree = 100 * (pol - pol_sum_tree) / pol;
  std::vector<double> average_key_difference((_keysetSize + 1), 0);
  for (int i = 0; i < (_keysetSize + 1); i++)
  {
    double result = static_cast<double>(sum_key_diff_matrice[i]) / key_difference_matrice.size();
    average_key_difference[i] = 100 * result / _G;
  };

  //------------------------ AVERAGE VALUE CALCULATION END ----------------------------------------------//

  outputFile << "PacketSize:" << _packetSize << "-" << "BufferSize:" << _G << "-" << "Attack Probability:" << attackProbability
             << "-" << "False Positivity Simple:" << fp_simple << "-" << "Pollution Number" << pol << "-" << "Detected Pollution Number" << pol_sum_simple << "-" << "SImple Ver Check Number:" << average_simple << std::endl;
  outputFile << "PacketSize:" << _packetSize << "-" << "BufferSize:" << _G << "-" << "Attack Probability:" << attackProbability
             << "-" << "False Positivity Tree:" << fp_tree << "-" << "-" << "Pollution Number" << pol << "Detected Pollution Number" << pol_sum_tree << "-" << "Tree VerResult:" << average_tree << std::endl;
  outputFile << "PacketSize:" << _packetSize << "-" << "BufferSize:" << _G << "-" << "Attack Probability:" << attackProbability
             << "-" << "False Positivity AR:" << fp_ar << "-" << "-" << "Pollution Number" << pol << "Detected Pollution Number" << pol_sum_ar << "-" << "AR Tree Ver Result:" << average_ar << std::endl;
  outputFile.flush();

  outputFile << std::endl;

  //}
  /////////////////////////----------------------------------------------------------------------------////////////////////////////////////////////////////
  outputFile_key_dist << "PacketSize:" << _packetSize << "-" << "BufferSize:" << _G
                      << "-" << "0 common:" << average_key_difference[0] << "-" << "1 common:" << average_key_difference[1] << "2 common:" << average_key_difference[2]
                      << "-" << "3 common:" << average_key_difference[3] << "4 common:" << average_key_difference[4] << std::endl;

  outputFile_key_dist << std::endl;

}; // closure of algorithm

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