#include <iostream>
#include <vector>
//#include <stdlib.h>
#include <algorithm>
#include "../networktopology.h"
#include <random>
//#include <unordered_map>
//#include <unordered_set>
#include "headers/hpacket.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/property_map.hpp>
#include <variant>


// Define the graph type using adjacency_list
//typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
  //      boost::property<boost::vertex_name_t, std::string>,
    //    boost::property<boost::vertex_list_t, std::vector<uint8_t>>,
      //  boost::property<boost::vertex_value_t, u_int8_t>
        //> Graph;

        // Define the custom struct to represent vertex properties
struct VertexProperties {
    int healthyReceived = 0;
    int pollutedReceived = 0;
    int pollutedDropped = 0;
    int totalNodeSend = 0;
    double falsePositiveEvents = 0.0;
    int checkNumber = 0;
    std::vector<std::vector<uint8_t>> keySet;
    std::string type = "Intermediate";
    double attackProbability = 0.0;
};typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexProperties> Graph;



int main() {
    // Create a graph object
    Graph g;

    // Add vertices to the graph and assign properties
    auto v0 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Source Node
    auto v1 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Intermediate_1 Node
    auto v2 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Compromised Node
    auto v3 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Intermediate_2 Node
    auto v4 = boost::add_vertex(VertexProperties{}, g); // Initialize properties for Destination Node

    // Create a connection between Nodes
    boost::add_edge(v0, v1, g);   // From Source to Intermediate_1 
    boost::add_edge(v0, v3, g);   // From Source to Intermediate_2 
    boost::add_edge(v0, v2, g);   // From Source to Adversary
    boost::add_edge(v1, v2, g);   // From Intermediate_1 to Adversary
    boost::add_edge(v2, v3, g);   // From Adversary to Intermediate
    boost::add_edge(v1, v3, g);   // From Intermediate_1 to Intermediate_2
    boost::add_edge(v3, v4, g);   // From  Intermediate_2 to Destination
    boost::add_edge(v1, v4, g);   // From  Intermediate_1 to Destination
    
    // Define paths
    std::vector<Graph::vertex_descriptor> path_1 = {v0, v1, v4};
    std::vector<Graph::vertex_descriptor> path_2 = {v0, v1, v2, v3, v4};
    std::vector<Graph::vertex_descriptor> path_3 = {v0, v1, v3, v4};
    std::vector<Graph::vertex_descriptor> path_4 = {v0, v2, v3, v4};
    std::vector<Graph::vertex_descriptor> path_5 = {v0, v3, v4};

    std::vector<std::vector<Graph::vertex_descriptor>>path_list;
    path_list.push_back(path_1); 
    path_list.push_back(path_2); 
    path_list.push_back(path_3); 
    path_list.push_back(path_4); 
    path_list.push_back(path_5); 
    
    // Access and manipulate vertex properties
    // Source Node
    g[v0].pollutedReceived = 0;
    g[v0].pollutedDropped = 0;
    g[v0].type = "Source";
    g[v0].totalNodeSend = 0;
    g[v0].attackProbability = 0;
    // Intermediate_1 Node
    g[v1].pollutedReceived = 0;
    g[v1].pollutedDropped = 0;
    g[v1].type = "Intermediate_1";
    g[v1].totalNodeSend = 0;
    g[v1].attackProbability = 0;
    // Adversary Node
    g[v2].pollutedReceived = 0;
    g[v2].pollutedDropped = 0;
    g[v2].type = "Adversary";
    g[v2].totalNodeSend = 0;
    g[v2].attackProbability = 0;
    // Intermediate_2 Node
    g[v3].pollutedReceived = 0;
    g[v3].pollutedDropped = 0;
    g[v3].type = "Intermediate";
    g[v3].totalNodeSend = 0;
    g[v3].attackProbability = 0;
    // Destination Node
    g[v4].pollutedReceived = 0;
    g[v4].pollutedDropped = 0;
    g[v4].type = "Destination";
    g[v4].totalNodeSend = 0;
    g[v4].attackProbability = 0;

    // Iterate over vertices and print their properties
    Graph::vertex_iterator vi, vend;
    for (boost::tie(vi, vend) = boost::vertices(g); vi != vend; ++vi) {
        std::cout << "Vertex: " << *vi << std::endl;
        std::cout << "Healthy Received: " << g[*vi].healthyReceived << std::endl;
        std::cout << "Polluted Received: " << g[*vi].pollutedReceived << std::endl;
        std::cout << "Polluted Dropped: " << g[*vi].pollutedDropped << std::endl;
        std::cout << "Total Node Send: " << g[*vi].totalNodeSend << std::endl;
        std::cout << "False Positive Events: " << g[*vi].falsePositiveEvents << std::endl;
        std::cout << "Check Number: " << g[*vi].checkNumber << std::endl;
        std::cout << "Type: " << g[*vi].type << std::endl;
        std::cout << "Attack Probability: " << g[*vi].attackProbability << std::endl;
        // Print keySet if needed
        std::cout << "Key Set:" << std::endl;
        for (const auto& row : g[*vi].keySet) {
            for (const auto& element : row) {
                std::cout << static_cast<int>(element) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}
