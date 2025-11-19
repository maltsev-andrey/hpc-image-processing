# Image Loader Utility
import os
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
#import time

class ImageLoader:
    """Handle loading and saving images for processing pipeline"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

    def get_image_paths(self, limit: Optional[int] = None) -> List[Path]:
        """Get list of all JPG images in dataset"""
        image_paths = sorted(self.dataset_path.glob("*.jpg"))

        if limit:
            image_paths = image_paths[:limit]

        print(f"Found {len(image_paths)} images")
        return image_paths

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load single image as NumPy array"""
        img = Image.open(image_path)
        # Convert to RGB if needed (some images might be grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img, dtype=np.float32)

    def save_image(self, image_array: np.ndarray, output_path: Path):
        """Save NumPy array as image"""
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Clip value to valid range and convert to uint8
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        img = Image.fromarray(image_array)
        img.save(output_path, quality=95)

    def load_batch(self, image_paths: List[Path], max_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """Load multiple images, optionally resizing"""
        images = []
        for path in image_paths:
            img = self.load_image(path)

            # Optional resize for testing with smaller images
            if max_size:
                img_pil = Image.fromarray(img.astype(np.uint8))
                img_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
                img = np.array(img_pil, dtype=np.float32)

            images.append(img)
        return images

if __name__=="__main__":
    # Test for loader
    print("Testing ImageLoader....")
    loader = ImageLoader("/nfs/shared/projects/image/datasets/coco/train2017")

    # Get first 5 images
    test_paths = loader.get_image_paths(limit=5)
    print(f"\nFirst image path: {test_paths[0]}")
    
    test_img = loader.load_image(test_paths[0])
    print(f"Image shape: {test_img.shape}")
    print(f"Image dtype: {test_img.dtype}")
    print(f"Value range [{test_img.min():.1f}, {test_img.max():.1f}]")       