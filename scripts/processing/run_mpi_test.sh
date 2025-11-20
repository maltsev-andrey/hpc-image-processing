#!/bin/bash
# Test MPI image processing

chmod +x "$0"

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
       --mca btl_tcp_if_include 10.10.10.0/24 \
       -np 5 \
       python3 /nfs/shared/projects/image/src/mpi/image_processor_mpi.py

echo ""
echo "Test complete!"
echo "Results in: results/gaussian_blur_mpi/"
