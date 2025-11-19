"""Gaussian Blur filter implementation"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple

class GaussianBlur:
    """Apply Gaussian blur to image"""

    def __init__(self, sigma: float=2.0):
        """
        Initialize Gaussian blur filter
        
        Args:
            sigma: Standard deviation for Gaussian kernel (higher = more blur)
        """
        self.sigma = sigma
        self.name = f"gaussian_blur_sigma{sigma}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to a single image
        Args -> image: Input image as numpy array (H, W, 3) with float32 values [0-255]
        Returns:   Blurred image as numpy array same shape as input
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), gor shape {image.shape}")

        # Apply Gaussian filter to each channel separately
        blurred = np.zeros_like(image)
        for channel in range(3):
            blurred[:, :, channel] = gaussian_filter(
                image[:, :, channel],
                sigma = self.sigma,
                mode = 'reflect' # Handle edges by reflection
            )

        return blurred

    def apply_batch(self, images : list) -> list:
        """
        Apply Gaussian blur to batch of images
        Args: images: List of numpy arrays
        Returns: List of blurred images
        """
        return [self.apply(img) for img in images]    

def create_gaussian_kernel(sigma: float, kernel_size: int = None) -> np.ndarray:
        """
        Create 2D Gaussian kernel for manual convolution (educational)
        Args:
            sigma: Standard deviation
            kernel_size: Size of kernel (if None, auto-calculate from sigma)
        Returns:
            2D Gaussian kernel normalized to sum to 1
        """
        if kernel_size is None:
            # Rule of thumb: kernel size = 2 * ceil(3 * sigma) + 1
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
    
        # Create  coordinate grids
        center = kernel_size // 2
        x,  y = np.mgrid[-center:center+1, -center:center+1]
    
        # 2D Gaussian formula    
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
        # Normalize so sum = 1
        kernel = kernel / kernel.sum()
    
        return kernel

if __name__ == "__main__":
    # Test the filter
    print("Testing Gaussian Blur filter...")

    from pathlib import Path
    import sys

    # Add parfetnt directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.utils.image_loader import ImageLoader
    from src.utils.timing import PerformanceTimer

    # Initialize
    loader = ImageLoader("/nfs/shared/projects/image/datasets/coco/train2017")
    blur = GaussianBlur(sigma=3.0)
    timer = PerformanceTimer()

    # Load test images
    print("\nLoading test images...")
    image_paths = loader.get_image_paths(limit=3)
    images = loader.load_batch(image_paths)

    print(f"Loaded {len(images)} images")
    print(f"First image shape: {images[0].shape}")

    # Apply blur with timing
    print(f"\nApplying {blur.name}...")
    blurred_images = []

    for i, img in enumerate(images):
        with timer.measure("gaussian_blur"):
            blurred = blur.apply(img)
            blurred_images.append(blurred)
        print(f"  Processed image {i + 1}/{len(images)}")

    # Show performance
    timer.print_summary()

    # Save results
    print("\nSaving results...")
    output_dir = Path("/nfs/shared/projects/image/results/gaussian_blur")

    for i, (original_path, blurred_img) in enumerate(zip(image_paths, blurred_images)):
        output_path = output_dir / f"blurred_{original_path.name}"
        loader.save_image(blurred_img, output_path)
        print(f"  Saved: {output_path.name}")

    print("\n Test complete")
    print(f"Results saved to: {output_dir}")

    # Show kernel info
    print(f"\nGaussian kernel info (sigma={blur.sigma}):")
    kernel = create_gaussian_kernel(blur.sigma)
    print(f"  Kernel size: {kernel.shape[0]}x{kernel.shape[1]}")
    print(f"  Center value: {kernel[kernel.shape[0]//2, kernel.shape[1]//2]:.6f}")

        