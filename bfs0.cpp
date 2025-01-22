#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include "helper.h"
#include <chrono>

std::vector<unsigned long long int> BFS0(
    const std::vector<std::vector<unsigned long long int>> edges,
    const std::vector<std::vector<unsigned long long int>> weights
) {

    size_t n = edges.size();


    std::vector<unsigned long long int> distances(
        n, std::numeric_limits<unsigned long long int>::max());


    using pii = std::pair<unsigned long long int, unsigned long long int>;
    std::priority_queue<pii, std::vector<pii>, std::greater<>> pq;


    distances[0] = 0;
    pq.push({0, 0}); 

    auto start_processing = std::chrono::high_resolution_clock::now();

    while (!pq.empty()) {
        unsigned long long int current_distance = pq.top().first;
        unsigned long long int current_node = pq.top().second;
        pq.pop();


        if (current_distance > distances[current_node]) continue;

        for (size_t i = 0; i < edges[current_node].size(); ++i) {
            unsigned long long int neighbor = edges[current_node][i];
            unsigned long long int weight = weights[current_node][i];

            if (distances[current_node] + weight < distances[neighbor]) {
                distances[neighbor] = distances[current_node] + weight;
                pq.push({distances[neighbor], neighbor});
            }
        }
    }

    auto end_processing = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> processing_time = end_processing - start_processing;
    std::cout << "Processing CPU time: " << processing_time.count() << "s" << std::endl;

    return distances;
}