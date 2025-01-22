#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "helper.h"

// Funkcja do tworzenia grafu
void create_graph(const std::string& filePath, unsigned long long int n, bool bidirectional,
                 std::vector<std::vector<unsigned long long int>>& edges,
                 std::vector<std::vector<unsigned long long int>>& weights) {
    // Otwieranie pliku
    std::ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Nie można otworzyć pliku: " + filePath);
    }

    std::string line;

    // Pomijanie pierwszych czterech linii pliku
    for (int i = 0; i < 4; ++i) {
        if (!std::getline(inputFile, line)) {
            throw std::runtime_error("Plik jest za krótki, brakuje danych po nagłówku.");
        }
    }

    // Inicjalizacja struktur grafu
    edges.resize(n);
    weights.resize(n);

    while (std::getline(inputFile, line)) {
        std::istringstream ss(line);
        unsigned long long int u, v;

        // Parsowanie dwóch wierzchołków
        if (!(ss >> u >> v)) {
            throw std::runtime_error("Nieprawidłowy format danych w pliku: " + line);
        }

        // Sprawdzanie zakresu wierzchołków
        if (u >= n || v >= n) {
            throw std::out_of_range("Wierzchołki wykraczają poza zakres: " + std::to_string(u) + ", " + std::to_string(v));
        }

        // Dodawanie krawędzi i wagi
        edges[u].push_back(v);
        weights[u].push_back(1);

        // Jeśli graf jest dwukierunkowy, dodaj krawędź w drugą stronę
        if (bidirectional) {
            edges[v].push_back(u);
            weights[v].push_back(1);
        }

    }

    // Zamknięcie pliku
    inputFile.close();
}