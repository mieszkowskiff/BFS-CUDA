#include <iostream>
#include "helper.h"
#include <vector>
#include <string>
int main() {


    std::vector<std::vector<unsigned long long int>> edges;
    std::vector<std::vector<unsigned long long int>> weights;
    std::string file_path;
    std::cout << "Enter the path to the file: ";
    std::cin >> file_path;
    unsigned long long int n;
    if (file_path == "journal") {
        n = 4847571;
    } else if (file_path == "small") {
        n = 258;
    } else if (file_path == "twitch") {
        n = 168114;
    } else {
        n = 168114;
    }
    create_graph(
        "./../datasets/" + file_path + ".txt", 
        n,
        false, 
        edges, 
        weights
        );
    std::cout << "Graph created" << std::endl;
    std::vector<unsigned long long int> bfs0_distances = BFS0(edges, weights);
    std::vector<unsigned long long int> bfs1_distances = BFS1(edges, weights);
    std::vector<unsigned long long int> bfs2_distances = BFS2(edges, weights);
    for (unsigned long long int i = 0; i < bfs0_distances.size(); ++i) {
        if (bfs0_distances[i] != bfs2_distances[i]) {
            std::cout << "BFS0 and BFS2 are not equal on index: " << i << std::endl;
            std::cout << "BFS0: " << bfs0_distances[i] << std::endl;
            std::cout << "BFS2: " << bfs2_distances[i] << std::endl;
            return 1;
        }
    }
    
    return 0;
}