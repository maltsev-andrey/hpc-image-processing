#!/bin/bash
# Test GPU image processing

PROJECT_ROOT="/nfs/shared/projects/image"

cd $PROJECT_ROOT

echo "=========================================="
echo "GPU Image Processing Test"
echo "=========================================="
echo "Device: srv-tesla-bme (Tesla P100)"
echo "Test size: 100 images"
echo ""

# Run on GPU host
cd $PROJECT_ROOT && python3 src/gpu/gaussian_blur_gpu.py

echo ""
echo "Test complete!"
