#include <iostream>
#include "helper.h"
#include <vector>


int main() {
    std::vector<std::vector<unsigned long long int>> edges = {
        {3},
        {2, 1},
        {0, 1, 3, 4},
        {0, 6},
        {5},
        {6, 4},
        {5, 3}
    };
    std::vector<std::vector<unsigned long long int>> weights = {
        {8},
        {3, 1},
        {20, 3, 10, 5},
        {8, 40},
        {10},
        {2, 10},
        {2, 40}
    };
    std::vector<unsigned long long int> distances = BFS2(edges, weights);
    for (unsigned long long int i = 0; i < distances.size(); i++) {
        std::cout << "Distance from 0 to " << i << " is " << distances[i] << std::endl;
    }
    return 0;
}