"""
Performance Benchmarking Suite for Input Manager Implementations

This module provides comprehensive benchmarking tools to compare the performance
between the C++/pybind11 implementation and the Python implementation of the
Input Manager. It measures frame processing latency, throughput, memory usage,
and provides detailed performance analysis.
"""

import sys
import os
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from contextlib import contextmanager
import json

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "InputManagerModule"))

from python_input_manager import PythonInputManager, InputManager as PythonCompatInputManager

# Try to import C++ implementation
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "InputManagerModule" / "build" / "lib"))
    import input_manager_native as cpp_im
    CPP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"C++ InputManager not available: {e}")
    CPP_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    implementation: str
    frame_count: int
    total_duration: float
    avg_fps: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    additional_metrics: Dict[str, Any]


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for Input Manager implementations.
    """
    
    def __init__(self, duration: int = 30, camera_index: int = 0):
        """
        Initialize benchmark suite.
        
        Args:
            duration: Duration of each benchmark test in seconds
            camera_index: Camera index to use for testing
        """
        self.duration = duration
        self.camera_index = camera_index
        self.results: List[BenchmarkResult] = []
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    @contextmanager
    def monitor_resources(self):
        """Context manager to monitor CPU and memory usage during benchmarks."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_samples = []
        
        # Start monitoring
        start_time = time.time()
        
        def sample_resources():
            try:
                cpu_percent = process.cpu_percent()
                if cpu_percent > 0:  # Ignore zero readings
                    cpu_samples.append(cpu_percent)
            except:
                pass
        
        try:
            yield sample_resources
        finally:
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            avg_cpu = np.mean(cpu_samples) if cpu_samples else 0
            memory_usage = max(final_memory - initial_memory, 0)
            
            # Store results in context
            self._last_cpu_usage = avg_cpu
            self._last_memory_usage = memory_usage
    
    def benchmark_python_implementation(self) -> BenchmarkResult:
        """
        Benchmark the Python Input Manager implementation.
        
        Returns:
            BenchmarkResult containing performance metrics
        """
        self.logger.info("Starting Python implementation benchmark...")
        
        frame_times = []
        processing_times = []
        successful_frames = 0
        total_attempts = 0
        
        with self.monitor_resources() as sample_fn:
            with PythonInputManager(
                camera_index=self.camera_index,
                enable_mediapipe=True
            ) as input_manager:
                
                if not input_manager.start_continuous_capture():
                    raise RuntimeError("Failed to start Python Input Manager")
                
                start_time = time.time()
                last_sample_time = start_time
                
                while time.time() - start_time < self.duration:
                    # Sample resources
                    sample_fn()
                    
                    # Measure frame processing
                    frame_start = time.time()
                    frame_data = input_manager.get_latest_frame()
                    frame_end = time.time()
                    
                    total_attempts += 1
                    
                    if frame_data is not None:
                        successful_frames += 1
                        frame_times.append(frame_end)
                        processing_times.append((frame_end - frame_start) * 1000)  # ms
                    
                    # Small sleep to prevent CPU spinning
                    time.sleep(0.001)
                
                input_manager.stop_continuous_capture()
        
        # Calculate metrics
        total_duration = time.time() - start_time
        success_rate = successful_frames / total_attempts if total_attempts > 0 else 0
        
        # Calculate FPS from frame intervals
        if len(frame_times) >= 2:
            frame_intervals = np.diff(frame_times)
            avg_fps = 1.0 / np.mean(frame_intervals) if np.mean(frame_intervals) > 0 else 0
        else:
            avg_fps = 0
        
        # Processing time statistics
        if processing_times:
            avg_latency = np.mean(processing_times)
            min_latency = np.min(processing_times)
            max_latency = np.max(processing_times)
        else:
            avg_latency = min_latency = max_latency = 0
        
        # Get Python-specific metrics
        performance_metrics = input_manager.get_performance_metrics()
        
        return BenchmarkResult(
            implementation="Python",
            frame_count=successful_frames,
            total_duration=total_duration,
            avg_fps=avg_fps,
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            memory_usage_mb=self._last_memory_usage,
            cpu_usage_percent=self._last_cpu_usage,
            success_rate=success_rate,
            additional_metrics=performance_metrics
        )
    
    def benchmark_cpp_implementation(self) -> Optional[BenchmarkResult]:
        """
        Benchmark the C++ Input Manager implementation if available.
        
        Returns:
            BenchmarkResult containing performance metrics or None if not available
        """
        if not CPP_AVAILABLE:
            self.logger.warning("C++ implementation not available for benchmarking")
            return None
        
        self.logger.info("Starting C++ implementation benchmark...")
        
        frame_times = []
        processing_times = []
        successful_frames = 0
        total_attempts = 0
        
        with self.monitor_resources() as sample_fn:
            input_manager = cpp_im.InputManager(self.camera_index)
            
            try:
                if not input_manager.start_continuous_capture():
                    raise RuntimeError("Failed to start C++ Input Manager")
                
                start_time = time.time()
                
                while time.time() - start_time < self.duration:
                    # Sample resources
                    sample_fn()
                    
                    # Measure frame processing
                    frame_start = time.time()
                    frame_data = input_manager.get_latest_frame()
                    frame_end = time.time()
                    
                    total_attempts += 1
                    
                    if frame_data is not None and hasattr(frame_data, 'get_frame'):
                        try:
                            frame_array = frame_data.get_frame()
                            if frame_array is not None and frame_array.size > 0:
                                successful_frames += 1
                                frame_times.append(frame_end)
                                processing_times.append((frame_end - frame_start) * 1000)  # ms
                        except:
                            pass
                    
                    # Small sleep to prevent CPU spinning
                    time.sleep(0.001)
                
            finally:
                input_manager.stop_continuous_capture()
        
        # Calculate metrics
        total_duration = time.time() - start_time
        success_rate = successful_frames / total_attempts if total_attempts > 0 else 0
        
        # Calculate FPS from frame intervals
        if len(frame_times) >= 2:
            frame_intervals = np.diff(frame_times)
            avg_fps = 1.0 / np.mean(frame_intervals) if np.mean(frame_intervals) > 0 else 0
        else:
            avg_fps = 0
        
        # Processing time statistics
        if processing_times:
            avg_latency = np.mean(processing_times)
            min_latency = np.min(processing_times)
            max_latency = np.max(processing_times)
        else:
            avg_latency = min_latency = max_latency = 0
        
        return BenchmarkResult(
            implementation="C++",
            frame_count=successful_frames,
            total_duration=total_duration,
            avg_fps=avg_fps,
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            memory_usage_mb=self._last_memory_usage,
            cpu_usage_percent=self._last_cpu_usage,
            success_rate=success_rate,
            additional_metrics={}
        )
    
    def run_comprehensive_benchmark(self) -> Dict[str, BenchmarkResult]:
        """
        Run comprehensive benchmarks for both implementations.
        
        Returns:
            Dictionary mapping implementation names to benchmark results
        """
        results = {}
        
        # Benchmark Python implementation
        try:
            python_result = self.benchmark_python_implementation()
            results["Python"] = python_result
            self.results.append(python_result)
        except Exception as e:
            self.logger.error(f"Python benchmark failed: {e}")
        
        # Small delay between tests
        time.sleep(2)
        
        # Benchmark C++ implementation if available
        if CPP_AVAILABLE:
            try:
                cpp_result = self.benchmark_cpp_implementation()
                if cpp_result:
                    results["C++"] = cpp_result
                    self.results.append(cpp_result)
            except Exception as e:
                self.logger.error(f"C++ benchmark failed: {e}")
        
        return results
    
    def generate_performance_report(self, results: Dict[str, BenchmarkResult], output_file: str = "performance_report.json") -> None:
        """
        Generate detailed performance report.
        
        Args:
            results: Benchmark results dictionary
            output_file: Output file path for JSON report
        """
        report = {
            "benchmark_config": {
                "duration_seconds": self.duration,
                "camera_index": self.camera_index,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": {},
            "comparison": {}
        }
        
        # Add detailed results
        for impl_name, result in results.items():
            report["results"][impl_name] = {
                "frame_count": result.frame_count,
                "total_duration": result.total_duration,
                "avg_fps": result.avg_fps,
                "avg_latency_ms": result.avg_latency_ms,
                "min_latency_ms": result.min_latency_ms,
                "max_latency_ms": result.max_latency_ms,
                "memory_usage_mb": result.memory_usage_mb,
                "cpu_usage_percent": result.cpu_usage_percent,
                "success_rate": result.success_rate,
                "additional_metrics": result.additional_metrics
            }
        
        # Generate comparison if both implementations available
        if "Python" in results and "C++" in results:
            python_result = results["Python"]
            cpp_result = results["C++"]
            
            report["comparison"] = {
                "fps_ratio_python_vs_cpp": python_result.avg_fps / cpp_result.avg_fps if cpp_result.avg_fps > 0 else 0,
                "latency_ratio_python_vs_cpp": python_result.avg_latency_ms / cpp_result.avg_latency_ms if cpp_result.avg_latency_ms > 0 else 0,
                "memory_difference_mb": python_result.memory_usage_mb - cpp_result.memory_usage_mb,
                "cpu_difference_percent": python_result.cpu_usage_percent - cpp_result.cpu_usage_percent,
                "success_rate_difference": python_result.success_rate - cpp_result.success_rate
            }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance report saved to {output_file}")
    
    def print_results_summary(self, results: Dict[str, BenchmarkResult]) -> None:
        """
        Print a summary of benchmark results to console.
        
        Args:
            results: Benchmark results dictionary
        """
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        for impl_name, result in results.items():
            print(f"\n{impl_name} Implementation:")
            print(f"  Frames Processed:     {result.frame_count}")
            print(f"  Duration:             {result.total_duration:.2f} seconds")
            print(f"  Average FPS:          {result.avg_fps:.2f}")
            print(f"  Average Latency:      {result.avg_latency_ms:.2f} ms")
            print(f"  Min Latency:          {result.min_latency_ms:.2f} ms")
            print(f"  Max Latency:          {result.max_latency_ms:.2f} ms")
            print(f"  Memory Usage:         {result.memory_usage_mb:.2f} MB")
            print(f"  CPU Usage:            {result.cpu_usage_percent:.2f}%")
            print(f"  Success Rate:         {result.success_rate*100:.2f}%")
        
        # Print comparison if both available
        if "Python" in results and "C++" in results:
            python_result = results["Python"]
            cpp_result = results["C++"]
            
            print(f"\nPERFORMANCE COMPARISON:")
            print(f"  Python vs C++ FPS:        {python_result.avg_fps:.2f} vs {cpp_result.avg_fps:.2f} ({python_result.avg_fps/cpp_result.avg_fps*100:.1f}%)")
            print(f"  Python vs C++ Latency:    {python_result.avg_latency_ms:.2f}ms vs {cpp_result.avg_latency_ms:.2f}ms")
            print(f"  Python vs C++ Memory:     {python_result.memory_usage_mb:.2f}MB vs {cpp_result.memory_usage_mb:.2f}MB")
            print(f"  Python vs C++ CPU:        {python_result.cpu_usage_percent:.2f}% vs {cpp_result.cpu_usage_percent:.2f}%")


def main():
    """Main function to run benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Input Manager Performance Benchmark")
    parser.add_argument("--duration", type=int, default=30, help="Benchmark duration in seconds")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--output", type=str, default="performance_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    # Run benchmarks
    benchmark = PerformanceBenchmark(duration=args.duration, camera_index=args.camera)
    
    print(f"Starting performance benchmarks (duration: {args.duration}s, camera: {args.camera})")
    print("Please ensure your camera is connected and no other applications are using it.")
    
    results = benchmark.run_comprehensive_benchmark()
    
    if results:
        benchmark.print_results_summary(results)
        benchmark.generate_performance_report(results, args.output)
    else:
        print("No benchmark results available. Please check camera connection and implementation availability.")


if __name__ == "__main__":
    main()