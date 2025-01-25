# Breadth First Search using CUDA programming

## About the project
This project is the part of 2024 / 2025 winter semester course in Graphic Processors in Computational Applications,
at Warsaw University of Technology, Faculty of Mathematics and Information Sciences.

## Get the dataset
As a dataset you can download [LiveJournal social network dataset](https://snap.stanford.edu/data/soc-LiveJournal1.html)
or [Twitter Interaction Network for the US Congress dataset](https://snap.stanford.edu/data/congress-twitter.html), or create your
own. Save it under ```./datasets```.

## Run the project
```{bash}
mkdir build
cd build
cmake ..
make
./BFS_CUDA
```

## Results
In case of big datasets (like journal of 4847571 nodes) GPU implementations are significantly faster:
```{bash}
Processing CPU time: 11.2739s
Time to copy data to GPU: 0.233658s
Time to process data on GPU: 0.0135745s
Time to copy data back to CPU: 0.0079864s
Total bfs1 time: 0.255219s
Time to copy data to GPU: 0.180219s
Time to process data on GPU: 0.156251s
Time to copy data back to CPU: 0.00731142s
Total bfs2 time: 0.343781s
```
Where ```bfs1``` and ```bfs2``` are two different GPU implementations.

## Issue
Unfortunately, the CPU implementations does not provide the same answers as CPU implementations. This issue is related to big graphs
only and I was unable to reproduce the problem on small, toy example.