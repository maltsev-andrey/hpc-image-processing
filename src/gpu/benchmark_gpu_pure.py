"""Benchmark pure GPU computation without I/O"""

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
import time


def benchmark_pure_gpu(num_images=1000, image_shape=(480, 640, 3), sigma=3.0):
    """
    Benchmark GPU with pre-loaded data (no disk I/O)
    
    Creates synthetic images in memory to isolate GPU performance
    """
    print("="*60)
    print("PURE GPU COMPUTATION BENCHMARK")
    print("="*60)
    print(f"Images: {num_images}")
    print(f"Shape: {image_shape}")
    print(f"Sigma: {sigma}")
    print()
    
    # Generate synthetic images in CPU memory
    print("Generating synthetic images in memory...")
    images = []
    for i in range(num_images):
        img = np.random.randint(0, 256, image_shape, dtype=np.uint8).astype(np.float32)
        images.append(img)
    print(f"Created {len(images)} synthetic images")
    print()
    
    # Warmup
    print("Warming up GPU...")
    img_gpu = cp.asarray(images[0])
    _ = gaussian_filter(img_gpu[:,:,0], sigma=sigma, mode='reflect')
    cp.cuda.Stream.null.synchronize()
    print()
    
    # Benchmark: Transfer + Process
    print("Benchmark 1: Transfer to GPU + Process + Transfer back")
    start = time.perf_counter()
    
    for img in images:
        # Transfer to GPU
        img_gpu = cp.asarray(img)
        
        # Process on GPU
        blurred_gpu = cp.zeros_like(img_gpu)
        for channel in range(3):
            blurred_gpu[:, :, channel] = gaussian_filter(
                img_gpu[:, :, channel],
                sigma=sigma,
                mode='reflect'
            )
        
        # Transfer back to CPU
        _ = cp.asnumpy(blurred_gpu)
    
    cp.cuda.Stream.null.synchronize()
    elapsed_transfer = time.perf_counter() - start
    throughput_transfer = num_images / elapsed_transfer
    
    print(f"  Time: {elapsed_transfer:.2f}s")
    print(f"  Throughput: {throughput_transfer:.2f} img/sec")
    print(f"  Time per image: {elapsed_transfer/num_images*1000:.2f}ms")
    print()
    
    # Benchmark: GPU-only processing (data stays on GPU)
    print("Benchmark 2: Pure GPU processing (no transfers)")
    
    # Pre-transfer all images to GPU
    print("  Pre-loading all images to GPU memory...")
    images_gpu = [cp.asarray(img) for img in images]
    cp.cuda.Stream.null.synchronize()
    
    start = time.perf_counter()
    
    for img_gpu in images_gpu:
        # Process on GPU only
        blurred_gpu = cp.zeros_like(img_gpu)
        for channel in range(3):
            blurred_gpu[:, :, channel] = gaussian_filter(
                img_gpu[:, :, channel],
                sigma=sigma,
                mode='reflect'
            )
    
    cp.cuda.Stream.null.synchronize()
    elapsed_gpu_only = time.perf_counter() - start
    throughput_gpu_only = num_images / elapsed_gpu_only
    
    print(f"  Time: {elapsed_gpu_only:.2f}s")
    print(f"  Throughput: {throughput_gpu_only:.2f} img/sec")
    print(f"  Time per image: {elapsed_gpu_only/num_images*1000:.2f}ms")
    print()
    
    # Analysis
    print("="*60)
    print("ANALYSIS")
    print("="*60)
    print(f"With transfers:    {throughput_transfer:.2f} img/sec")
    print(f"GPU only:          {throughput_gpu_only:.2f} img/sec")
    print(f"Transfer overhead: {(1 - throughput_transfer/throughput_gpu_only)*100:.1f}%")
    print()
    print(f"Speedup vs CPU single (36.63 img/sec):")
    print(f"  With transfers: {throughput_transfer/36.63:.2f}x")
    print(f"  GPU only:       {throughput_gpu_only/36.63:.2f}x")
    print()
    print(f"Speedup vs MPI cluster (141.13 img/sec):")
    print(f"  With transfers: {throughput_transfer/141.13:.2f}x")
    print(f"  GPU only:       {throughput_gpu_only/141.13:.2f}x")
    
    # GPU memory info
    mempool = cp.get_default_memory_pool()
    print()
    print(f"GPU Memory Used: {mempool.used_bytes()/(1024**2):.2f} MB")


if __name__ == "__main__":
    # Test with different sizes
    for num_images in [100, 500, 1000, 5000]:
        benchmark_pure_gpu(num_images=num_images)
        print("\n" + "="*60 + "\n")
