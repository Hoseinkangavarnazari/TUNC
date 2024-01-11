#ifndef NODE_H
#define NODE_H
#include <vector>
#include <iostream>
#include "ff.h"
#include <iostream>
#include <vector>

#include <iostream>
#include <vector>
#include <cstdint> // for uint8_t
#include "hpacket.h"   

// Define a struct to represent the input type
struct NodeInput {
    std::vector<uint8_t> vectorInput;
    std::vector<std::vector<uint8_t>> matrixInput;
    uint8_t integerInput;
};

class Graph {
public:

    int numNodes; // Number of nodes in the graph
    std::vector<NodeInput> values;  // Values of each node
    std::vector<NodeInput> outputs; // Outputs for each node
    std::vector<std::vector<int>> adjacencyList; // Adjacency list to store connections

    Graph(int numNodes) : numNodes(numNodes) {
        // Resize the vectors to accommodate the given number of nodes
        values.resize(numNodes, NodeInput{}); // Initialize all values to default NodeInput
        outputs.resize(numNodes, NodeInput{}); // Initialize outputs to default NodeInput
    }

    // Function to set the input values of a node
    void setInput(int node, const NodeInput& input) ;

    // Function to set the value of a node
    void setValue(int node, const NodeInput& value);
    // Function to add a directed edge from node1 to node2
    void addDirectedEdge(int node1, int node2);

    // Function to perform operations in the network
    void performOperations() ;

    // Function to print the outputs of each node
    void printOutputs() ;

private:
    
};

#endif 