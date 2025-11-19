# Project 2: Distributed Image Processing Pipeline

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
- [ ] **Phase 2**: MPI distribution across compute nodes
- [ ] **Phase 3**: GPU acceleration with CUDA
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
- **Head node**: srv-hpc-01 (6 cores)
- **Compute nodes**: srv-hpc-02 to 05 (24 cores total)
- **Network**: 10.10.10.0/24 internal cluster network

### GPU Host
- **Node**: srv-tesla-bme
- **GPU**: NVIDIA Tesla P100
- **CUDA cores**: 3584
- **Memory**: 16GB HBM2

## Performance Goals

### Phase 1 (CPU Baseline)
- [x] Single-threaded: ~36 images/sec ✓

### Phase 2 (MPI Cluster)
- [ ] Target: 150-200 images/sec (5-node cluster)

### Phase 3 (GPU Acceleration)  
- [ ] Target: 2000+ images/sec (50-100x speedup)

### Phase 4 (Hybrid Pipeline)
- [ ] Target: Optimal routing based on filter complexity

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
