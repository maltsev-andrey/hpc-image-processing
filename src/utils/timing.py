# Timing utility
import time
from contextlib import contextmanager
from typing import Dict
from pathlib import Path
import json

class PerformanceTimer:
    """Track performance metrics for image processing"""

    def __init__(self):
        self.metrics: Dict[str, list] = {}

    @contextmanager
    def measure(self, operation_name: str):
        """Context manager for timing operations"""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            self.metrics[operation_name].append(elapsed)

    def get_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        if operation_name not in self.metrics:
                return {}
            
        times = self.metrics[operation_name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'images_per_second': len(times) / sum(times) if sum(times) > 0 else 0       
        }

    def print_summary(self):
        """Print sumary of all measurements"""
        print("\n=== Performance Summary ===")
        for op_name in sorted(self.metrics.keys()):
            stats = self.get_stats(op_name)
            print(f"\n{op_name}")
            print(f"  Images processed: {stats['count']}")
            print(f"  Total time: {stats['total']:.2f}s")
            print(f"  Mean time: {stats['mean']*1000:.2f}ms per image")
            print(f"  Throughput: {stats['images_per_second']:.2f} images/sec")
    
    def save_to_file(self, filepath: Path):
        """Save metrics to JSON file"""
        summary = {op: self.get_stats(op) for op in self.metrics.keys()}
        with open(filepath, 'w') as f:
            json.dump(summary, indent=2, fp=f)
        print(f"Metrics saved to {filepath}")

if __name__=="__main__":
    # Test for timer
    print("Testing PerformanceTimer...")
    timer = PerformanceTimer()
    
    # Simulate some processing
    for i in range(5):
        with timer.measure("test_operation"):
            time.sleep(0.1) # Simulate work
    
    timer.print_summary()        




















            