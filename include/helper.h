#ifndef HELPER_H
#define HELPER_H
 
#include <vector>
#include <string>

std::vector<unsigned long long int> BFS0(
    std::vector<std::vector<unsigned long long int>> edges,
    std::vector<std::vector<unsigned long long int>> weights
);


std::vector<unsigned long long int> BFS1(
    std::vector<std::vector<unsigned long long int>> edges,
    std::vector<std::vector<unsigned long long int>> weights
);

std::vector<unsigned long long int> BFS2(
    std::vector<std::vector<unsigned long long int>> edges,
    std::vector<std::vector<unsigned long long int>> weights
);

void create_graph(const std::string& filePath, unsigned long long int n, bool bidirectional,
                 std::vector<std::vector<unsigned long long int>>& edges,
                 std::vector<std::vector<unsigned long long int>>& weights);

#endif