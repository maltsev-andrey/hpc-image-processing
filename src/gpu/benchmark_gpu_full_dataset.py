"""Benchmark GPU on full COCO dataset"""

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.image_loader import ImageLoader
from src.utils.timing import PerformanceTimer

class GaussianBlurGPUOptimized:
    """Optimized GPU processing for large datasets"""

    def __init__(self, sigma: float = 3.0, batch_size: int = 100):
        self.sigma = sigma
        self.batch_size = batch_size

        cp.cuda.Device(0).use()
        self.device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()

    def process_images(self, image_paths, output_dir, save_results = True):
        """
        Process large dataset with progress reporting
        
        Args:
            image_paths: List of image paths to process
            output_dir: Where to save results
            save_results: Whether to save images (set False for pure benchmark)
        """
        loader = ImageLoader(str(image_paths[0].parent))
        timer = PerformanceTimer()

        total_images = len(image_paths)
        processed = 0
        errors = 0

        print(f"Processing {total_images} images...")
        print(f"Batch size: {self.batch_size}")
        print(f"Save results: {save_results}")
        print()

        overall_start = time.perf_counter()

        # Process in batches
        for batch_start in range(0, total_images, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_images)
            batch_paths = image_paths[batch_start:batch_end]

            try:
                # Load batch
                with timer.measure("load"):
                    images = [loader.load_image(p) for p in batch_paths]

                # Process on GPU
                with timer.measure("gpu_process"):
                    blurred_images = []

                    for img in images:
                        # Transfer to GPU
                        img_gpu = cp.asarray(img)

                        # Process
                        blurred_gpu = cp.zeros_like(img_gpu)
                        for channel in range(3):
                            blurred_gpu[:, :, channel] = gaussian_filter(
                                img_gpu[:, :, channel],
                                sigma=self.sigma,
                                mode='reflect'
                            )

                        # Transfer to back
                        blurred = cp.asnumpy(blurred_gpu)
                        blurred_images.append(blurred)

                # Optional save results
                if save_results:
                    with timer.measure("save"):
                        for i, (path, blurred) in enumerate(zip(batch_paths, blurred_images)):
                            output_path = output_dir / f"gpu_{path.name}"
                            loader.save_image(blurred, output_path)

                processed += len(batch_paths)

                # Progress report every 1000 images
                if processed % 1000 == 0:
                    elapsed = time.perf_counter() - overall_start
                    current_throughput = processed / elapsed
                    eta_seconds = (total_images - processed) / current_throughput

                    print(f"Progress: {processed}/{total_images} images "
                          f"({processed/total_images*100:.1f}%) - "
                          f"{current_throughput:.2f} img/sec - "
                          f"ETA: {eta_seconds/60:.1f} min")

            except Exception as e:
                print(f"Error in batch {batch_start}-{batch_end}: {e}")
                errors += len(batch_paths)
                continue

        overall_elapsed = time.perf_counter() - overall_start
        
        return {
            'total_images': total_images,
            'processed': processed,
            'errors': errors,
            'total_time': overall_elapsed,
            'throughput': processed / overall_elapsed,
            'load_stats': timer.get_stats('load'),
            'process_stats': timer.get_stats('gpu_process'),
            'save_stats': timer.get_stats('save') if save_results else None,
        }

def main():
    """Run full dataset benchmark"""

    #dataset_path = "/nfs/shared/projects/image/datasets/coco/train2017"
    #output_dir = Path("/nfs/shared/projects/image/results/gaussian_blur_gpu_full")

    # LOCAL dataset on Tesla SSD (fast!)
    dataset_path = "/mnt/old_nfs/projects/image/datasets/coco/train2017/"
    
    # Results can go to local or HPC NFS
    output_dir = Path("/tmp/gpu_results")  # Local temp (benchmark only)
    # OR save to HPC NFS:
    # output_dir = Path("/mnt/hpc_nfs/projects/image/results/gaussian_blur_gpu_full")

    print("="*70)
    print("GPU FULL DATASET BENCHMARK")
    print("="*70)
    print(f"Dataset: COCO train2017")
    print(f"Output: {output_dir}")
    print()

    # Initialize
    blur_gpu = GaussianBlurGPUOptimized(sigma = 3.0, batch_size=100)
    loader = ImageLoader(dataset_path)

    print(f"GPU Device: {blur_gpu.device_name}")
    print()

    # Get all image paths
    print("Loading image paths...")
    image_paths = loader.get_image_paths(limit=None) # All images
    print(f"Found {len(image_paths)} images")
    print()

    # Ask iser about saving
    print("Options:")
    print("1. Process with saving results (slower, ~400-500 img/sec)")
    print("2. Process without saving (faster, ~900 img/sec, benchmark only)")
    print()

    # For automated run, choose option 1 (with saving)
    save_results = False
    print("Running with saving enabled...")
    print()

    # Process
    start_time = time.time()
    results = blur_gpu.process_images(image_paths, output_dir, save_results= save_results)

    # Summary
    print()
    print("="*60)
    print("FULL DATASET RESULTS")
    print("="*60)
    print(f"Total images: {results['total_images']}")
    print(f"Processed: {results['processed']}")
    print(f"Errors: {results['errors']}")
    print(f"Total time: {results['total_time']:.2f}s ({results['total_time']/60:.2f} min)")
    print(f"Throughput: {results['throughput']:.2f} img/sec")
    print()

    print("Performance Breakdown:")
    if results['load_stats']:
        load_throughput = results['load_stats']['count'] / results['load_stats']['total']
        print(f"  Loading: {load_throughput:.2f} img/sec")
    
    if results['process_stats']:
        process_throughput = results['process_stats']['count'] / results['process_stats']['total']
        print(f"  GPU Processing: {process_throughput:.2f} img/sec")
    
    if results['save_stats']:
        save_throughput = results['save_stats']['count'] / results['save_stats']['total']
        print(f"  Saving: {save_throughput:.2f} img/sec")
    
    print()
    print("Comparison with Previous Phases:")
    print(f"  Phase 1 (CPU single): 36.63 img/sec")
    print(f"  Phase 2 (MPI cluster): 141.13 img/sec")
    print(f"  Phase 3 (GPU): {results['throughput']:.2f} img/sec")
    print()
    print(f"  Speedup vs Phase 1: {results['throughput']/36.63:.2f}x")
    print(f"  Speedup vs Phase 2: {results['throughput']/141.13:.2f}x")
    print()
    
    # GPU memory
    mempool = cp.get_default_memory_pool()
    print(f"GPU Memory Used: {mempool.used_bytes()/(1024**2):.2f} MB")
    print()
    print("="*70)
    print(f"Benchmark completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
    









                

                        
                



















        