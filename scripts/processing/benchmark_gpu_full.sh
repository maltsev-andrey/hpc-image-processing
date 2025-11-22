#!/bin/bash
# Benchmark GPU on full COCO dataset

chmod +x "$0"

PROJECT_ROOT="/nfs/shared/projects/image"

echo "=========================================="
echo "GPU Full Dataset Benchmark"
echo "=========================================="
echo "Dataset: COCO train2017 (118,287 images)"
echo "Device: srv-tesla-bme (Tesla P100)"
echo ""
echo "Expected time: ~4-5 minutes"
echo "Expected throughput: 400-500 img/sec"
echo ""

cd $PROJECT_ROOT
python3 src/gpu/benchmark_gpu_full_dataset.py | tee logs/gpu_full_benchmark_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Benchmark complete!"
