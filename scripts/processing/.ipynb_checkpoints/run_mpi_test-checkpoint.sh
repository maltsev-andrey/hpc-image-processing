#!/bin/bash
# Test MPI image processing

PROJECT_ROOT="/nfs/shared/projects/image"
HOSTFILE="/nfs/shared/cluster_config/hostfile_cpu"

cd $PROJECT_ROOT

echo "Starting MPI Image Processing Test"
echo "==================================="
echo "Nodes: 5 (srv-hpc-01 to srv-hpc-05)"
echo "Test size: 500 images"
echo ""

mpirun --hostfile $HOSTFILE \
       --map-by node \
       -np 5 \
       python3 src/mpi/image_processor_mpi.py

echo ""
echo "Test complete!"
echo "Results in: results/gaussian_blur_mpi/"