# Project 2: Distributed Image Processing Pipeline

## Overview
Hybrid CPU+GPU image processing system for HPC cluster

## Architecture
- Head Node (srv-hpc-01): Coordination, NFS storage
- Compute Nodes (srv-hpc-02 to 05): CPU-based filtering
- GPU Node (srv-tesla-bme): CUDA-accelerated heavy filters

## Dataset
COCO train2017: ~118,000 images (18GB)

## Filters Implemented
1. Gaussian Blur (CPU + GPU)
2. Sobel Edge Detection (CPU + GPU)
3. Histogram Equalization (CPU)
4. Median Filter (CPU)

## Development Phases
- Phase 1: Single-node Python implementation ✓
- Phase 2: MPI distribution across compute nodes
- Phase 3: GPU acceleration with CUDA
- Phase 4: Hybrid CPU+GPU pipeline
- Phase 5: Benchmarking and optimization

## Usage
See scripts/processing/ for execution scripts
