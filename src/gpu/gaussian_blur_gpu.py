"""GPU-accelerated Gaussian Blur using CuPy"""

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
from typing import List
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class GaussianBlurGPU:
    """Apply Gaussian blur using GPU acceleration"""

    def __init__(self, sigma: float=2.0, device_id: int=0):
        """
        Initialize GPU-accelerated Gaussian blur
        
        Args:
            sigma: Standard deviation for Gaussian kernel
            device_id: GPU device ID (default: 0)
        """
        self.sigma = sigma
        self.device_id = device_id
        self.name = f"gaussian_blur_gpu_sigma{sigma}"

        # Set GPU device
        cp.cuda.Device(device_id).use()

        # Get GPU info
        self.device_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode()

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to single image on GPU
        
        Args:
            image: Input image as numpy array (H, W, 3) with float32 values [0-255]
            
        Returns:
            Blurred image as numpy array
        """
        if image.ndim !=3 or image.shape[2] !=3:
            raise ValueError(f"Expected RGB image (H,W,3), got shape {image.shape}")

        #Transfer to GPU
        image_gpu = cp.asarray(image)

        # Apply Gaussian filter to each channel
        blurred_gpu = cp.zeros_like(image_gpu)
        for channel in range(3):
            blurred_gpu[:, :, channel] = gaussian_filter(
                image_gpu[:, :, channel],
                sigma = self.sigma,
                mode='reflect'
            )
        # Transfer back to CPU
        blurred = cp.asnumpy(blurred_gpu)

        return blurred

    def apply_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply Gaussian blur to batch of images on GPU
        
        Args: images: List of numpy arrays
        Returns: List of blurred images
        """
        return [self.apply(img) for img in images] 

    def apply_batch_optimized(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Optimized batch processing - keeps data on GPU longer
        
        Args:
            images: List of numpy arrays
            
        Returns:
            List of blurred images
        """
        blurred_images = []

        for image in images:
            # Transfer to GPU
            image_gpu = cp.asarray(image)

            # Process on GPU
            blurred_gpu = cp.zeros_like(image_gpu)
            for channel in range(3):
                blurred_gpu[:, :, channel] = gaussian_filter(
                    image_gpu[:, :, channel],
                    sigma = self.sigma,
                    mode = 'reflect'
                )

            # Transfer back to CPU
            blurred = cp.asnumpy(blurred_gpu)
            blurred_images.append(blurred)

        return blurred_images

    def get_memory_info(self) -> dict:
        """Get GPU memory usage information"""
        mempool = cp.get_default_memory_pool()
        return {
            'used bytes': mempool.used_bytes(),
            'total_bytes': mempool.total_bytes(),
            'used_mb': mempool.used_bytes() / (1024**2),
            'total_mb': mempool.total_bytes() / (1024**2),
        }

if __name__ == "__main__":
    # Test GPU filter
    print("Testing GPU Gaussian Blur...")

    from src.utils.image_loader import ImageLoader
    from src.utils.timing import PerformanceTimer

    # Initialize
    loader = ImageLoader("/nfs/shared/projects/image/datasets/coco/train2017")
    blur_gpu = GaussianBlurGPU(sigma=3.0)
    timer = PerformanceTimer()

    print(f"GPU Device: {blur_gpu.device_name}")
    print(f"Filter: {blur_gpu.name}")

    # Load test image
    print("\nLoading test images...")
    image_paths = loader.get_image_paths(limit=100)
    images = loader.load_batch(image_paths)

    print(f"Loaded {len(images)} images")
    print(f"First image shape: {images[0].shape}")

    # Warmup GPU (first run is slower)
    print("\nWarming up GPU...")
    _=blur_gpu.apply(images[0])

    # Process batch with timing
    print(f"\nProcessing {len(images)} images on GPU...")
    blurred_images = []

    for i, img in enumerate(images):
        with timer.measure("gpu_gaussian_blur"):
            blurred = blur_gpu.apply(img)
            blurred_images.append(blurred)

        if  (i+1) % 20 ==0:
            print(f"  Process {i+1}/{len(images)} images")

    # Show performance
    timer.print_summary()

    # Memory info
    print("\nGPU Memory Usage:")
    mem_info = blur_gpu.get_memory_info()
    print(f" Used: {mem_info['used_mb']:.2f} MB")
    print(f" Total: {mem_info['total_mb']:2f} MB")

    # Save output results
    print("\nSaving sample results...")
    output_dir = Path("/nfs/shared/projects/image/results/gaussian_blur_gpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(5, len(blurred_images))):
        output_path = output_dir /f"gpu_blurred_{image_paths[i].name}"
        loader.save_image(blurred_images[i], output_path)

    print(f"\n GPU test complete!")
    print(f"Sample results saved to: {output_dir}")
    