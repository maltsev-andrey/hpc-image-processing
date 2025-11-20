"""MPI-based distributed image processing"""

from mpi4py import MPI
import numpy as np
from pathlib import Path
from typing import List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.image_loader import ImageLoader
from src.utils.timing import PerformanceTimer
from src.filters.gaussian_blur import GaussianBlur

class MPIImageProcessor:
    """Distribute image processing across MPI cluster"""

    def __init__(self, dataset_path: str, output_dir: str):
        """
        Initialize MPI processor
        
        Args:
            dataset_path: Path to input images
            output_dir: Directory for processed images
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)

        # Initialize utilities on all ranks
        self.loader = ImageLoader(str(self.dataset_path))
        self.timer = PerformanceTimer()

        # Master rank info
        self.is_master = (self.rank == 0)

        if self.is_master:
            print(f"MPI Image Processor initialized")
            print(f"  Cluster size: {self.size} nodes")
            print(f"  Dataset: {self.dataset_path}")
            print(f"  Output: {self.output_dir}")

    def distribute_work(self, image_paths: List[Path]) -> List[Path]:
        """
        Distribute image paths across MPI ranks
        
        Args:
            image_paths: List of all image paths (only used by master)
            
        Returns:
            List of image paths assigned to this rank
        """
        if self.is_master:
            # Master: split paths into equal batches
            total_images = len(image_paths)
            batch_size = total_images // self.size
            remainder = total_images % self.size

            batches = []
            start_idx = 0

            for rank in range(self.size):
                # Distribute remainder images to first ranks
                end_idx = start_idx + batch_size + (1 if rank < remainder else 0)
                batch = image_paths[start_idx:end_idx]
                batches.append(batch)
                start_idx = end_idx

            print(f"\nWork distribution:")
            for rank, batch in enumerate(batches):
                print(f"  Rank {rank}: {len(batch)} images")
            print()
        else:
            batches = None

        # Scatter batches to all ranks
        my_batch = self.comm.scatter(batches, root=0)

        return my_batch

    def process_batch(self, image_paths: List[Path], filter_obj, 
                     filter_name: str) -> dict:
        """
        Process assigned batch of images with given filter
        
        Args:
            image_paths: Paths to images for this rank
            filter_obj: Filter object with apply() method
            filter_name: Name for timing metrics
            
        Returns:
            Dictionary with processing statistics
        """
        processed_count = 0
        
        for img_path in image_paths:
            try:
                # Load image
                with self.timer.measure(f"{filter_name}_load"):
                    image = self.loader.load_image(img_path)
                
                # Apply filter
                with self.timer.measure(f"{filter_name}_process"):
                    filtered = filter_obj.apply(image)
                
                # Save result
                with self.timer.measure(f"{filter_name}_save"):
                    output_path = self.output_dir / f"rank{self.rank}_{img_path.name}"
                    self.loader.save_image(filtered, output_path)
                
                processed_count += 1
                
                # Progress reporting (every 100 images)
                if processed_count % 100 == 0:
                    print(f"Rank {self.rank}: Processed {processed_count}/{len(image_paths)} images")
            
            except Exception as e:
                print(f"Rank {self.rank}: Error processing {img_path.name}: {e}")
                continue

        # Collect statistics
        stats = {
            'rank': self.rank,
            'processed': processed_count,
            'total_assigned': len(image_paths),
            'load_stats': self.timer.get_stats(f"{filter_name}_load"),
            'process_stats': self.timer.get_stats(f"{filter_name}_process"),
            'save_stats': self.timer.get_stats(f"{filter_name}_save"),
        }

        return stats

    def gather_results(self, local_stats: dict) -> Optional[List[dict]]:
        """
        Gather statistics from all ranks to master
        
        Args:
            local_stats: Statistics from this rank
            
        Returns:
            List of all statistics (only on master rank)
        """
        all_stats = self.comm.gather(local_stats, root=0)
        return all_stats

    def print_summary(self, all_stats: List[dict]):
        """Print processing summary (master only)"""
        if not self.is_master:
            return

        print("\n" + "="*60)
        print("MPI PROCESSING SUMMARY")
        print("="*60)

        total_processed = sum(s['processed'] for s in all_stats)
        total_assigned = sum(s['total_assigned'] for s in all_stats)

        print(f"\nTotal images processed: {total_processed} / {total_assigned}")
        print(f"Cluster nodes: {self.size}")

        print("\nPer-node statistics:")
        for stats in all_stats:
            rank = stats['rank']
            processed = stats['processed']

            # Calculate throughput
            if stats['process_stats']:
                proc_time = stats['process_stats']['total']
                throughput = processed / proc_time if proc_time > 0 else 0
                print(f"  Rank {rank}: {processed} images, {throughput:.2f} img/sec")

        # Calculate overall throughput
        print("\nOverall performance:")
        max_process_time = max(
            s['process_stats']['total']
            for s in all_stats
            if s['process_stats']
        )

        overall_throughput = total_processed / max_process_time if max_process_time > 0 else 0
        print(f"  Total throughput: {overall_throughput:.2f} images/sec")
        print(f"  Speedup vs single-node: {overall_throughput / 36.63:.2f}x")
        
        print("="*60 + "\n")

def main():
    """Test MPI image processor"""

    # Configuration
    dataset_path = "/nfs/shared/projects/image/datasets/coco/train2017"
    output_dir = "/nfs/shared/projects/image/results/gaussian_blur_mpi"
    num_images = 500  # Test with 500 images first

    # Initialize MPI processor
    processor = MPIImageProcessor(dataset_path, output_dir)            

    # Master loads image list
    if processor.is_master:
        print(f"\nLoafing image paths (limit: {num_images})...")
        image_paths = processor.loader.get_image_paths(limit=num_images)
    else:
        image_paths = None

    # Disftibutr work
    my_batch = processor.distribute_work(image_paths)

    print(f"Rank {processor.rank}: Received {len(my_batch)} images to process")

    # Create filter
    blur_filter = GaussianBlur(sigma=3.0)

    # Process batch
    print(f"Rank {processor.rank}: Starting processing...")
    local_stats = processor.process_batch(my_batch, blur_filter, "gaussian_blur")

    print(f"Rank {processor.rank}: Processing complete - {local_stats['processed']} images")

    # Gather results
    all_stats = processor.gather_results(local_stats)

    # Master prints summary
    if processor.is_master:
        processor.print_summary(all_stats)

if __name__ == "__main__":
    main()
            
    




















        