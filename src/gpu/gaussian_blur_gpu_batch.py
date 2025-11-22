"""Optimized GPU batch processing - minimal CPU↔GPU transfers"""

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
from typing import List
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class GaussianBlurGPUBatch:
    """Optimized batch processing on GPU"""

    def __init__(self, sigma: float=2.0, batch_size: int = 50):
        """
        Initialize GPU batch processor
        
        Args:
            sigma: Standard deviation for Gaussian kernel
            batch_size: Number of images to process together on GPU
        """
        self.sigma = sigma
        self.batch_size = batch_size
        self.name = f"gaussian_blur_gpu_batch{batch_size}_sigma{sigma}"

        # Set GPU device
        cp.cuda.Device(0).use()

        # Get GPU info
        self.device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()

    def process_batch_gpu(self, images_gpu: List[cp.ndarray]) -> List[cp.ndarray]:
        """
        Process batch of images already on GPU
        
        Args:
            images_gpu: List of CuPy arrays on GPU
            
        Returns:
            List of blurred CuPy arrays on GPU
        """
        blurred_batch = []

        for image_gpu in images_gpu:
            blurred_gpu = cp.zeros_like(image_gpu)
            for channel in range(3):
                blurred_gpu[:, :, channel] = gaussian_filter(
                    image_gpu[:, :, channel],
                    sigma=self.sigma,
                    mode='reflect'
                )
            blurred_batch.append(blurred_gpu)
        
        return blurred_batch

    def apply_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process batch with optimized GPU memory transfers
        Strategy: Transfer batch → Process all on GPU → Transfer back
        
        Args: images: List of numpy arrays (CPU)
        Returns: List of blurred numpy arrays (CPU)
        """
        all_blurred = []

        # Process in chunks of batch_size
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            # Single transfer: CPU → GPU
            images_gpu = [cp.asarray(img) for img in batch]
            
            # Process entire batch on GPU (no transfers)
            blurred_gpu = self.process_batch_gpu(images_gpu)
            
            # Single transfer: GPU → CPU
            blurred_batch = [cp.asnumpy(img) for img in blurred_gpu]
            
            all_blurred.extend(blurred_batch)
        
        return all_blurred
 
    def get_memory_info(self) -> dict:
        """Get GPU memory usage information"""
        mempool = cp.get_default_memory_pool()
        return {
            'used_mb': mempool.used_bytes() / (1024**2),
            'total_mb': mempool.total_bytes() / (1024**2),
        }

if __name__ == "__main__":
    from src.utils.image_loader import ImageLoader
    from src.utils.timing import PerformanceTimer

    print("Testing Optimized GPU Batch Processing...")
    
    # Initialize
    loader = ImageLoader("/nfs/shared/projects/image/datasets/coco/train2017")
    blur_gpu = GaussianBlurGPUBatch(sigma=3.0, batch_size=50)
    timer = PerformanceTimer()

    print(f"GPU Device: {blur_gpu.device_name}")
    print(f"Batch size: {blur_gpu.batch_size}")

    # Load test image
    print("\nLoading test images...")
    image_paths = loader.get_image_paths(limit=100)
    images = loader.load_batch(image_paths)

    # Test with different batch sizes
    for num_images in [100, 500, 1000]:
        print(f"\n{'='*60}")
        print(f"Testing with {num_images} images")
        print('='*60)

        # Load images
        image_paths = loader.get_image_paths(limit=num_images)
        images = loader.load_batch(image_paths)

        # Warmup
        if num_images == 100:
            _=blur_gpu.apply_batch([images[0]])

        # Process with timing
        with timer.measure(f"gpu_batch_{num_images}"):
            blurred = blur_gpu.apply_batch(images)

        stats = timer.get_stats(f"gpu_batch_{num_images}")
        throughput = num_images / stats['total']
    
        print(f"Processed: {num_images} images")
        print(f"Total time: {stats['total']:.2f}s")
        print(f"Throughput: {throughput:.2f} img/sec")
        print(f"Speedup vs CPU single: {throughput/36.63:.2f}x")
        print(f"Speedup vs MPI cluster: {throughput/141.13:.2f}x")
        
        # Memory
        mem = blur_gpu.get_memory_info()
        print(f"GPU Memory: {mem['used_mb']:.2f} MB used")
    
    print(f"\n Optimized batch test complete!")
    