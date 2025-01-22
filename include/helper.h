#ifndef HELPER_H
#define HELPER_H

#include <vector>

std::vector<unsigned long long int> BFS1(
    std::vector<std::vector<unsigned long long int>> edges,
    std::vector<std::vector<unsigned long long int>> weights
);

std::vector<unsigned long long int> BFS2(
    std::vector<std::vector<unsigned long long int>> edges,
    std::vector<std::vector<unsigned long long int>> weights
);

#endif