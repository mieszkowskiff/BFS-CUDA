#ifndef HELPER_H
#define HELPER_H

#include <vector>

std::vector<unsigned long long int> BFS(
    std::vector<std::vector<unsigned long long int>> edges,
    std::vector<std::vector<unsigned long long int>> weights
);

#endif