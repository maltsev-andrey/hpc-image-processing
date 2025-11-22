# Distributed Image Processing Pipeline

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://python.org)
[![MPI](https://img.shields.io/badge/MPI-OpenMPI-green)](https://www.open-mpi.org/)
[![Cores](https://img.shields.io/badge/CPU_Cores-48-red)](https://github.com)
[![Performance](https://img.shields.io/badge/Performance-252M_updates/sec-orange)](https://github.com)

High-performance image processing system for HPC cluster with hybrid CPU+GPU architecture.

## Overview

This project shows parallel image processing using MPI-based cluster computing and CUDA GPU acceleration. The system processes large-scale image datasets (COCO train2017 with 118,287 images) applying various computational filters.

## Architecture

- **Head Node** (srv-hpc-01): Task coordination, NFS storage (400GB)
- **Compute Nodes** (srv-hpc-02 to 05): CPU-based parallel processing (48 cores total)
- **GPU Node** (srv-tesla-bme): CUDA acceleration with Tesla P100 (3584 cores)
- **Network**: Dual-network topology (external + internal 10Gbps)

## Dataset

**COCO train2017**: 118,287 images (~18GB)
- Source: http://cocodataset.org
- Format: JPEG, variable resolutions
- Storage: `/nfs/shared/projects/image/datasets/coco/train2017/`

## Implemented Filters

### 1. Gaussian Blur (Phase 1 - CPU baseline)
- **Algorithm**: Convolution with Gaussian kernel
- **Implementation**: SciPy `gaussian_filter`
- **Performance**: ~36 images/sec (single-threaded, sigma=3.0)
- **Use case**: Noise reduction, image smoothing

### Coming Soon
- Sobel Edge Detection
- Histogram Equalization
- Median Filter

## Project Structure
```
image/
├── datasets/
│   └── coco/
│       └── train2017/          # 118K images
├── src/
│   ├── utils/
│   │   ├── image_loader.py     # Image I/O utilities
│   │   └── timing.py           # Performance metrics
│   └── filters/
│       └── gaussian_blur.py    # Gaussian blur implementation
├── results/
│   └── gaussian_blur/          # Processed images
├── scripts/
│   ├── setup/
│   └── processing/
└── docs/
```

## Development Phases

- [x] **Phase 1**: Single-node CPU implementation (baseline)
- [x] **Phase 2**: MPI distribution across compute nodes
- [x] **Phase 3**: GPU acceleration with CUDA
- [ ] **Phase 4**: Hybrid CPU+GPU pipeline
- [ ] **Phase 5**: Benchmarking and optimization

## Current Status: Phase 1 Complete

### Baseline Performance (CPU, single-threaded)
- **Filter**: Gaussian Blur (sigma=3.0)
- **Throughput**: 36.63 images/sec
- **Mean processing time**: 27.3ms per image
- **Test images**: 640x480 RGB

### Infrastructure
- NFS shared storage: 40GB used / 400GB total
- Python modules with proper package structure
- Image loader with batch processing support
- Performance timing utilities

## Technology Stack

### Current (Phase 1)
- **Python 3.9+**: Core development
- **NumPy**: Array operations
- **SciPy**: Filter implementations
- **Pillow (PIL)**: Image I/O

### Planned (Phase 2-4)
- **mpi4py**: Distributed computing
- **PyCUDA / CuPy**: GPU acceleration
- **OpenMPI**: Cluster communication

## Usage

### Run Gaussian Blur test
```bash
cd /nfs/shared/projects/image
python3 src/filters/gaussian_blur.py
```

Output: Processed images in `results/gaussian_blur/`

### Test utilities independently
```bash
# Test image loader
python3 src/utils/image_loader.py

# Test performance timer
python3 src/utils/timing.py
```

## Installation

### Prerequisites
- Red Hat Enterprise Linux 9.5
- Python 3.9+
- NFS shared storage mounted at `/nfs/shared`

### Python dependencies
```bash
pip3 install numpy scipy pillow
```

## Development Workflow

All code is developed as `.py` modules (not Jupyter notebooks) for:
- Clean version control
- Professional code structure  
- Easy execution from terminal
- Reusable components

## Hardware Configuration

### HPC Cluster
- **Head node**: srv-hpc-01 (8 cores)
- **Compute nodes**: srv-hpc-02 to 05 (48 cores total)
- **Network**: 10.10.10.0/24 internal cluster network

### GPU Host
- **Node**: srv-tesla-bme
- **GPU**: NVIDIA Tesla P100
- **CUDA cores**: 3584
- **Memory**: 16GB HBM2

## Performance Goals

###[x] Phase 1: (Single-node CPU implementation)
- Single-threaded: ~36 images/sec 
- Completed: November 18, 2025

###[x] Phase 2: (MPI distribution across compute nodes)
- Performance: 141.13 images/sec
- Speedup: 3.85x (77% parallel efficiency)
- Full COCO dataset: 37 minutes
- Completed: November 20, 2025

###[x] Phase 3: GPU acceleration with CUDA ✓
  - Performance: 155.07 images/sec (local SSD)
  - Speedup: 4.23x vs CPU, 1.10x vs MPI cluster
  - GPU: NVIDIA Tesla P100 (CuPy implementation)
  - Processing time: 12.71 minutes for full dataset
  - Completed: November 22, 2025

###[ ] **Phase 4**: Hybrid CPU+GPU pipeline
###[ ] **Phase 5**: Benchmarking and optimization

## Current Status: Phase 2 Complete 

### Phase 2 Results

**MPI Cluster Performance** (5 nodes):
- **Throughput**: 141.13 images/sec
- **Speedup**: 3.85x vs single-node
- **Parallel Efficiency**: 77% (excellent for shared storage)
- **Full Dataset Processing**: 118,287 images in 37 minutes

**Load Balancing**: Near-perfect distribution
- All nodes: 28-29 images/sec
- Variance: <2% between nodes

See detailed analysis: [Phase 2 Performance Report](docs/PHASE2_PERFORMANCE_REPORT.md)

## Current Status: Phase 3 Complete

### Phase 3 Results

**GPU Acceleration Performance** (Tesla P100):

**Configuration Testing**:
- **3a. GPU + NFS/Powerline**: 77 img/sec (network bottleneck - 50 Mbps)
- **3b. GPU + Local NVMe SSD**: 155.07 img/sec (I/O + transfer bottleneck)
- **3c. GPU Pure Computation**: 458-915 img/sec (theoretical maximum)

**Processing Time**: 12.71 minutes for 118,287 images

**Bottleneck Analysis**:
- Network (powerline): Limited to 77 img/sec (GPU 0% utilized)
- Local SSD: Achieved 155.07 img/sec (GPU 20-53% utilized)
- Pure GPU capability: 915 img/sec (when data pre-loaded in GPU memory)

**Key Finding**: Real-world GPU performance (155 img/sec) achieves only 16.9% of theoretical GPU capability (915 img/sec), limited by:
1. Sequential I/O processing (not batch processing)
2. CPU↔GPU transfer overhead (~35% of time)
3. JPEG decoding overhead (~30% of time)
4. Small image sizes underutilize 3,584 CUDA cores

**GPU Utilization**: 20-53% (I/O bound, not compute bound)

See detailed analysis: [Phase 3 Performance Report](docs/PHASE3_PERFORMANCE_REPORT.md)

### Comprehensive Performance Comparison

| Phase | Configuration       | Throughput     | Speedup | Efficiency | Bottleneck         |
|-------|---------------------|----------------|---------|------------|--------------------|
| 1     | CPU single          | 36.63 img/sec  | 1.0x    | -          | CPU compute        |
| 2     | MPI (5 nodes)       | 141.13 img/sec | 3.85x   | 77%        | NFS I/O contention |
| 3a    | GPU + NFS/Powerline | 77 img/sec     | 2.10x   | -          | Network 50Mbps     |
| 3b    | GPU + Local SSD     | 155.07 img/sec | 4.23x   | 16.9%      | I/O + transfers    |
| 3c    | GPU pure            | 915 img/sec    | 25x     | -          | None (theoretical) |

**Interesting Result**: GPU with local SSD (155 img/sec) is only 10% faster than 5-node MPI cluster (141 img/sec), showing that distributed CPU computing can compete with single GPU when I/O is the bottleneck.

## Future Enhancements

1. **Additional Filters**
   - Edge detection (Sobel, Canny)
   - Color correction
   - Sharpening, median filtering

2. **Advanced Features**
   - Dynamic load balancing
   - Pipeline optimization
   - Real-time processing monitoring

3. **Production Features**
   - REST API for job submission
   - Web interface for result visualization
   - Automated benchmarking suite

## Learning Objectives

- Domain decomposition strategies
- MPI communication patterns
- GPU memory management
- CPU vs GPU performance analysis
- Hybrid computing architectures
- Large-scale data processing

## References

- COCO Dataset: https://cocodataset.org
- Gaussian Filter Theory: https://en.wikipedia.org/wiki/Gaussian_blur
- Related: Project 1 - Heat Equation Solver (MPI parallelization)

## Author

Andrey Maltsev - HPC Cluster Infrastructure & Scientific Computing

## License

Educational project for learning distributed systems and GPU computing.
