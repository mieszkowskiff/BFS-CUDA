#include <iostream>
#include "helper.h"
#include <vector>


int main() {
    std::vector<std::vector<unsigned long long int>> edges = {
        {2, 3},
        {2, 1},
        {0, 1, 3, 4},
        {0, 2, 6},
        {2, 5},
        {6, 4},
        {5, 3}
    };
    std::vector<std::vector<unsigned long long int>> weights = {
        {20, 8},
        {3, 1},
        {20, 3, 10, 5},
        {8, 10, 40},
        {5, 10},
        {2, 10},
        {2, 40}
    };
    std::vector<unsigned long long int> distances = BFS(edges, weights);
    for (unsigned long long int i = 0; i < distances.size(); i++) {
        std::cout << "Distance from 0 to " << i << " is " << distances[i] << std::endl;
    }
    return 0;
}