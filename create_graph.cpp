#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "helper.h"


void create_graph(const std::string& filePath, unsigned long long int n, bool bidirectional,
                 std::vector<std::vector<unsigned long long int>>& edges,
                 std::vector<std::vector<unsigned long long int>>& weights) {

    std::ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Can't open file: " + filePath);
    }

    std::string line;

    for (int i = 0; i < 4; ++i) {
        if (!std::getline(inputFile, line)) {
            throw std::runtime_error("The file is to short.");
        }
    }

    edges.resize(n);
    weights.resize(n);

    while (std::getline(inputFile, line)) {
        std::istringstream ss(line);
        unsigned long long int u, v;

        if (!(ss >> u >> v)) {
            throw std::runtime_error("Invalid file format: " + line);
        }

        if (u >= n || v >= n) {
            throw std::out_of_range("List of vertecies is to long: " + std::to_string(u) + ", " + std::to_string(v));
        }

        edges[u].push_back(v);
        weights[u].push_back(1);

        if (bidirectional) {
            edges[v].push_back(u);
            weights[v].push_back(1);
        }

    }

    inputFile.close();
}