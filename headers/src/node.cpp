#include <vector>
#include <iostream>
#include "../ff.h"
#include <iostream>
#include <vector>
#include "../node.h"
#include <cstdint> // for uint8_t
#include "../hpacket.h"
#include "../rlnc_decoder.h"

  // Include your tud.h header


std::vector<uint8_t> node_codedSymbol;
std::vector<uint8_t> node_codedSymbol2;
std::vector<uint8_t> node_MAC;
std::vector<std::vector<uint8_t>> node_publickeyset;
std::vector<uint8_t> node_privateKey;
int node_number_of_mac;
std::vector<uint8_t> node_coefficientvector;
int numNodes;
int node_generationSize;int node_symbolSize;FieldSize node_fieldSize;
int nodeIndex;

////
// Define a struct to represent the input type
//struct NodeInput {
    std::vector<std::vector<std::vector<uint8_t>>> dataInput;  // 3D input matrice for incoming data packets to nodes
    std::vector<std::vector<std::vector<uint8_t>>> nodeOutput;  // 3D input matrice for outputs of nodes
   // dataInput.resize(numNodes,std::vector<std::vector<uint8_t>>(node_generationSize,std::vector<uint8_t>(node_symbolSize, uint8_t 0)));
    std::vector<std::vector<std::vector<uint8_t>>> keyInput;  // 3D input matrice for assigned keys  to nodes
   // std::vector<std::vector<uint8_t>> matrixInput;
   // uint8_t integerInput;
    std::vector<std::vector<int>> adjacencyList; // Adjacency list to store connections
    std::vector<NodeInput> values;  // Values of each node
//};
/////

hpacket hh(node_codedSymbol, node_MAC,node_publickeyset, node_privateKey, node_number_of_mac, node_coefficientvector);
rlnc_decoder decoder(node_generationSize, node_symbolSize, node_fieldSize);


//class Graph {
//public:
 //   Graph(int numNodes) {
        // Resize the vectors to accommodate the given number of nodes
      //  dataInput.resize(numNodes, NodeInput{}); // Initialize all values to default NodeInput
      //  nodeOutput.resize(numNodes, NodeInput{}); // Initialize outputs to default NodeInput
  // }

    // Function to set the value of a node
   // void setValue(int node, const NodeInput& value) {
  //      values[node] = value;
 //   }

    // Function to add a directed edge from node1 to node2
    void addDirectedEdge(int node1, int node2) {
        adjacencyList[node1].push_back(node2);
    

    // Function to set the input values of a node
    //void setInput(int nodeIndex) {
        dataInput[node1]=nodeOutput[node2];
    }


    // Function to perform operations in the network
    void performOperations(int numNodes) {
        // Perform some operations using values of Node 0
        // You can access vectorInput, matrixInput, and integerInput as needed
        // Source Node operations
        for (int nodeindex = 0; nodeindex < numNodes; nodeindex++)
        { 
        if (nodeIndex == 0)
        {
            hh.macCalculator();
            hh.signCalculator();
            hh.packetAppender();
        }
         // Intermediate Node operations
        else if (0 < nodeIndex < numNodes-1) {
            hh.Verifier();
            hh.packetCombiner();
        }
        // Sink Node operations
        else if (nodeIndex == numNodes-1) {
            hh.Verifier();
            decoder.decode();
        }
        }
      
    }

        ////////////
        
      //  uint8_t sum = values[0].integerInput;
       // hh.macCalculator();
        //hh.signCalculator();
        //hh.packetAppender();
        //for (const auto& val : values[0].vectorInput) {
          //  sum += val;
        //}

        // Send the result to Node 1
      //  for (const auto& neighbor : adjacencyList[0]) {
          //  outputs[neighbor].integerInput = sum;
        //}

        // Call the function from tud.h in Node 2
        // Note: Modify the following line based on the actual content of tud.h
        // std::vector<uint8_t> resultVector = computeVector(values[2]);
        // std::vector<std::vector<uint8_t>> resultMatrix = computeMatrix(values[2]);

        // Assuming computeVector and computeMatrix functions in tud.h
     //   std::vector<uint8_t> resultVector = hh.macCalculator();
      //  std::vector<std::vector<uint8_t>> resultMatrix = computeMatrix(values[2].integerInput);

    //    outputs[2].vectorInput = resultVector;
      //  outputs[2].matrixInput = resultMatrix;

        // Note: Adjust the code based on the actual content and signature of the functions in tud.h
   // };

    // Function to print the outputs of each node
//    void printOutputs() {
  //      for (int i = 0; i < numNodes; ++i) {
  //          std::cout << "Output for node " << i << ": ";
  //          std::cout << "Vector: ";
   //         for (const auto& val : outputs[i].vectorInput) {
   //             std::cout << static_cast<int>(val) << " ";
    //        }
    //        std::cout << "Matrix: ";
    //        for (const auto& row : outputs[i].matrixInput) {
   //             for (const auto& val : row) {
   //                 std::cout << static_cast<int>(val) << " ";
   //             }
   //             std::cout << "| ";
  //          }
  //          std::cout << "Integer: " << static_cast<int>(outputs[i].integerInput);
  //          std::cout << std::endl;
  ///      }
 //   }

//private:
   // int numNodes; // Number of nodes in the graph
 //   std::vector<NodeInput> values;  // Values of each node
  //  std::vector<NodeInput> outputs; // Outputs for each node
 //   std::vector<std::vector<int>> adjacencyList; // Adjacency list to store connections
//};

//int main() {
    // Create a directed graph with 4 nodes
  //  Graph myGraph(4);

    // Set inputs for Node 0, Node 1, and Node 2 with different types of data
  //  NodeInput input0 = {{1, 2, 3}, {{4, 5}, {6, 7}}, 8};
  //  NodeInput input1 = {{9, 10}, {{11, 12}, {13, 14}}, 15};
 //   NodeInput input2 = {{16, 17, 18}, {{19, 20}, {21, 22}}, 23};

 //   myGraph.setInput(0, input0);
  //  myGraph.setInput(1, input1);
  //  myGraph.setInput(2, input2);

    // Add directed edges between nodes
 //   myGraph.addDirectedEdge(0, 1);
  //  myGraph.addDirectedEdge(0, 2);

    // Perform operations in the network
  //  myGraph.performOperations();

    // Print the outputs
  //  myGraph.printOutputs();

  //  return 0;
//};
