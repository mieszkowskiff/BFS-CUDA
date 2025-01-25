#include "helper.h"
#include <stdio.h>
#include <vector>
#include <limits>
#include <iostream>
#include <chrono>

#define CUDA_CHECK(cudaStatus)                                      \
    if(cudaStatus != cudaSuccess)                                   \
        std::cout << cudaGetErrorString(cudaStatus) << std::endl;   \


__global__ void BFS_step2(
    unsigned long long int* d_edges,
    unsigned long long int* d_indices,
    unsigned long long int* d_weights,
    unsigned long long int* d_distances,
    unsigned long long int* d_frontier,
    unsigned long long int* d_frontier_size,
    int* d_is_in_frontier
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= *d_frontier_size) {
        return;
    }

    unsigned long long int nodeA = d_frontier[id];

    __syncthreads();
    *d_frontier_size = 0;
    __syncthreads();

    for (unsigned long long int i = d_indices[nodeA]; i < d_indices[nodeA + 1]; i++) {
        unsigned long long int nodeB = d_edges[i];
        unsigned long long int weight = d_weights[i];
        unsigned long long int old_distance = d_distances[nodeB];
        unsigned long long int new_distance = d_distances[nodeA] + weight;
        if (new_distance < old_distance) {
            atomicMin(&d_distances[nodeB], new_distance);
            unsigned long long int is_in_frontier = atomicCAS(&d_is_in_frontier[nodeB], 0, 1);
            if (is_in_frontier == 0) {
                unsigned long long int new_frontier_index = atomicAdd(d_frontier_size, 1);
                d_frontier[new_frontier_index] = nodeB;
            }
        }
    }   
}


std::vector<unsigned long long int> BFS2(
    std::vector<std::vector<unsigned long long int>> edges,
    std::vector<std::vector<unsigned long long int>> weights
) {

    std::vector<unsigned long long int> h_edges;
    std::vector<unsigned long long int> h_indices;
    std::vector<unsigned long long int> h_weights;

    // flatten edges and weights
    unsigned long long int current_index = 0;
    for (const auto& inner : edges) {
        h_indices.push_back(current_index);
        h_edges.insert(h_edges.end(), inner.begin(), inner.end());
        current_index += inner.size();
    }
    for (const auto& inner : weights) {
        h_weights.insert(h_weights.end(), inner.begin(), inner.end());
    }
    unsigned long long int n = edges.size();
    unsigned long long int m = h_edges.size();
    h_indices.push_back(m);
    

    unsigned long long int* d_edges;
    unsigned long long int* d_indices;
    unsigned long long int* d_weights;


    // initialize distances
    unsigned long long int* h_distances = new unsigned long long int[n];
    for (unsigned long long int i = 0; i < n; i++) {
        h_distances[i] = std::numeric_limits<unsigned long long int>::max();
    }
    h_distances[0] = 0;

    unsigned long long int* d_distances;
    
    // initialize frontier
    unsigned long long int* d_frontier;

    unsigned long long int frontier_size = 1;
    unsigned long long int* d_frontier_size;
    
    int* is_in_frontier;

    auto start_copying = std::chrono::high_resolution_clock::now();

    // copy data to GPU
    CUDA_CHECK(cudaMalloc(&d_edges, h_edges.size() * sizeof(unsigned long long int)));
    CUDA_CHECK(cudaMalloc(&d_indices, h_indices.size() * sizeof(unsigned long long int)));
    CUDA_CHECK(cudaMalloc(&d_weights, h_weights.size() * sizeof(unsigned long long int)));

    CUDA_CHECK(cudaMemcpy(
        d_edges, 
        h_edges.data(), 
        h_edges.size() * sizeof(unsigned long long int), 
        cudaMemcpyHostToDevice
        ));
    CUDA_CHECK(cudaMemcpy(
        d_indices, 
        h_indices.data(), 
        h_indices.size() * sizeof(unsigned long long int), 
        cudaMemcpyHostToDevice
        ));
    CUDA_CHECK(cudaMemcpy(
        d_weights, 
        h_weights.data(), 
        h_weights.size() * sizeof(unsigned long long int), 
        cudaMemcpyHostToDevice
        ));

    CUDA_CHECK(cudaMalloc(&d_distances, n * sizeof(unsigned long long int)));
    CUDA_CHECK(cudaMemcpy(d_distances, h_distances, n * sizeof(unsigned long long int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_frontier, n * sizeof(unsigned long long int)));
    CUDA_CHECK(cudaMemset(d_frontier, 0, n * sizeof(unsigned long long int)));

    CUDA_CHECK(cudaMalloc(&d_frontier_size, sizeof(unsigned long long int)));
    CUDA_CHECK(cudaMemcpy(d_frontier_size, &frontier_size, sizeof(unsigned long long int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&is_in_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(is_in_frontier, 0, n * sizeof(int)));

    auto end_copying = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_copying = end_copying - start_copying;
    std::cout << "Time to copy data to GPU: " << elapsed_seconds_copying.count() << "s\n";

    auto start_processing = std::chrono::high_resolution_clock::now();

    while(true) {
        CUDA_CHECK(cudaMemcpy(&frontier_size, d_frontier_size, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        if (frontier_size == 0) {
            break;
        }

        //bfs step
        CUDA_CHECK(cudaMemset(is_in_frontier, 0, n * sizeof(int)));
        BFS_step2<<<(frontier_size + 255) / 256, 256>>>(
            d_edges,
            d_indices,
            d_weights,
            d_distances,
            d_frontier,
            d_frontier_size,
            is_in_frontier
        );
    //endloop
    CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end_processing = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_processing = end_processing - start_processing;
    std::cout << "Time to process data on GPU: " << elapsed_seconds_processing.count() << "s\n";

    auto start_copying_back = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(h_distances, d_distances, n * sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

    auto end_copying_back = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds_copying_back = end_copying_back - start_copying_back;
    std::cout << "Time to copy data back to CPU: " << elapsed_seconds_copying_back.count() << "s\n";
    std::cout << "Total bfs2 time: " << 
        elapsed_seconds_copying.count() + 
        elapsed_seconds_processing.count() + 
        elapsed_seconds_copying_back.count() << "s\n";
    
    std::vector<unsigned long long int> distances(h_distances, h_distances + n);

    cudaFree(d_edges);
    cudaFree(d_indices);
    cudaFree(d_weights);
    cudaFree(d_distances);
    cudaFree(d_frontier);
    cudaFree(d_frontier_size);
    cudaFree(is_in_frontier);

    return distances;

}