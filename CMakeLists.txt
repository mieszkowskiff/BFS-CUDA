cmake_minimum_required(VERSION 3.18) # Wymagane dla CUDA i nowoczesnego CMake
project(BFS_CUDA LANGUAGES CXX CUDA)

# Ustaw wersję standardu C++ i CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Źródła projektu
set(CPP_SOURCES
    main.cpp
    bfs0.cpp
)
 
set(CUDA_SOURCES
    bfs1.cu
    bfs2.cu
    create_graph.cpp
)

message(STATUS "CPP Sources: ${CPP_SOURCES}")

# Dodaj katalogi z nagłówkami
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


# Utwórz plik wykonywalny z plików C++ i CUDA
add_executable(BFS_CUDA ${CPP_SOURCES} ${CUDA_SOURCES})


# Ustawienie flag kompilatora dla CUDA (opcjonalne)
set_target_properties(BFS_CUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON # Umożliwia oddzielną kompilację plików CUDA
)

# Opcjonalne flagi do debugowania i optymalizacji
target_compile_options(BFS_CUDA PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
