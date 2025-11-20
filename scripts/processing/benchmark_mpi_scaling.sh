#!/bin/bash
# Comprehensive MPI scaling benchmark

PROJECT_ROOT="/nfs/shared/projects/image"
HOSTFILE="/nfs/shared/cluster_config/hostfile_cpu"
RESULTS_FILE="$PROJECT_ROOT/docs/phase2_benchmark_results.txt"

cd $PROJECT_ROOT

# Clear previous results
> $RESULTS_FILE

echo "======================================" | tee -a $RESULTS_FILE
echo "MPI Image Processing Scaling Benchmark" | tee -a $RESULTS_FILE
echo "======================================" | tee -a $RESULTS_FILE
echo "Date: $(date)" | tee -a $RESULTS_FILE
echo "Cluster: 5 nodes (srv-hpc-01 to srv-hpc-05)" | tee -a $RESULTS_FILE
echo "Filter: Gaussian Blur (sigma=3.0)" | tee -a $RESULTS_FILE
echo "Single-node baseline: 36.63 img/sec" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# Test with different image counts
for NUM_IMAGES in 500 1000 5000 10000 50000 118287; do
    echo "========================================" | tee -a $RESULTS_FILE
    echo "Testing with $NUM_IMAGES images" | tee -a $RESULTS_FILE
    echo "========================================" | tee -a $RESULTS_FILE
    echo "Start time: $(date)" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE
    
    OUTPUT_DIR="$PROJECT_ROOT/results/gaussian_blur_mpi_${NUM_IMAGES}"
    
    # Run MPI processing
    mpirun --hostfile $HOSTFILE \
           --map-by node \
           --mca btl_tcp_if_include 10.10.10.0/24 \
           -np 5 \
           python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.mpi.image_processor_mpi import MPIImageProcessor
from src.filters.gaussian_blur import GaussianBlur

processor = MPIImageProcessor(
    '$PROJECT_ROOT/datasets/coco/train2017',
    '$OUTPUT_DIR'
)

if processor.is_master:
    image_paths = processor.loader.get_image_paths(limit=$NUM_IMAGES if $NUM_IMAGES < 118287 else None)
else:
    image_paths = None

my_batch = processor.distribute_work(image_paths)
blur_filter = GaussianBlur(sigma=3.0)
local_stats = processor.process_batch(my_batch, blur_filter, 'gaussian_blur')
all_stats = processor.gather_results(local_stats)

if processor.is_master:
    processor.print_summary(all_stats)
" | tee -a $RESULTS_FILE
    
    echo "" | tee -a $RESULTS_FILE
    echo "End time: $(date)" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE
    
    # Short pause between tests
    sleep 2
done

echo "======================================" | tee -a $RESULTS_FILE
echo "Benchmark Complete!" | tee -a $RESULTS_FILE
echo "Results saved to: $RESULTS_FILE" | tee -a $RESULTS_FILE
echo "======================================" | tee -a $RESULTS_FILE
