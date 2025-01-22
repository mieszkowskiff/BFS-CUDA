#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include "helper.h"

// Funkcja BFS0
std::vector<unsigned long long int> BFS0(
    const std::vector<std::vector<unsigned long long int>> edges,
    const std::vector<std::vector<unsigned long long int>> weights
) {
    // Liczba wierzchołków w grafie
    size_t n = edges.size();

    // Wektor odległości, inicjalizowany na maksymalną wartość
    std::vector<unsigned long long int> distances(
        n, std::numeric_limits<unsigned long long int>::max());

    // Kolejka priorytetowa do przechowywania wierzchołków do odwiedzenia
    using pii = std::pair<unsigned long long int, unsigned long long int>;
    std::priority_queue<pii, std::vector<pii>, std::greater<>> pq;

    // Odległość od węzła 0 do samego siebie wynosi 0
    distances[0] = 0;
    pq.push({0, 0}); // (odległość, wierzchołek)

    while (!pq.empty()) {
        unsigned long long int current_distance = pq.top().first;
        unsigned long long int current_node = pq.top().second;
        pq.pop();

        // Jeśli znaleźliśmy lepszą ścieżkę wcześniej, ignorujemy
        if (current_distance > distances[current_node]) continue;

        // Przeglądamy sąsiadów aktualnego węzła
        for (size_t i = 0; i < edges[current_node].size(); ++i) {
            unsigned long long int neighbor = edges[current_node][i];
            unsigned long long int weight = weights[current_node][i];

            // Relaksacja krawędzi: sprawdzamy, czy możemy poprawić odległość
            if (distances[current_node] + weight < distances[neighbor]) {
                distances[neighbor] = distances[current_node] + weight;
                pq.push({distances[neighbor], neighbor});
            }
        }
    }

    return distances;
}