import time
import psutil
import numpy as np
from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import gc


class ModelComparison:
    def __init__(self):
        # Simulate Raspberry Pi 3B+ constraints
        # 1GB RAM, 4 cores @ 1.4GHz
        torch.set_num_threads(4)

        # Initialize base models first
        self.yolo8_base = YOLO("../model/yolov8x.pt")
        self.yolo9_base = YOLO("../model/yolov9c.pt")

        # Now load the best.pt weights to both models
        self.yolo8_best = YOLO("../model/best.pt")
        self.yolo9_best = YOLO("../model/best.pt")

    def get_model_size(self, model):
        """
        Calculate approximate size of model in memory.
        """
        param_size = 0
        for param in model.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = (param_size + buffer_size) / (1024 * 1024)
        return total_size

    def benchmark_inference(self, model, image, num_iterations=50):
        """Benchmark inference performance using time-based measurements"""
        # First let's do a warm-up run
        for _ in range(5):
            _ = model(image)

        # Clear any GPU cache and run garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Measure inference times
        inference_times = []
        for _ in range(num_iterations):
            # Time the inference
            start_time = time.time()
            _ = model(image)
            inference_time = (
                time.time() - start_time
            ) * 1000  # Convert to milliseconds
            inference_times.append(inference_time)

            # Don't need to sleep here as we're just measuring raw inference time

        # Calculate statistics
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        model_size = self.get_model_size(model)

        return {
            "avg_inference_time_ms": avg_time,
            "std_inference_time_ms": std_time,
            "model_size_mb": model_size,
        }

    def benchmark_model_with_cpu(self, model, image):
        """Use a more direct method to measure CPU usage"""
        # Get the baseline CPU usage
        psutil.cpu_percent(interval=0.1)  # First call to reset
        time.sleep(0.2)  # Wait to get a clean measurement

        start_cpu = psutil.cpu_percent(percpu=True)
        baseline_cpu_avg = np.mean(start_cpu)

        # Run inference in a tight loop to maximize CPU effect
        start_time = time.time()
        iterations = 0

        # Run for exactly 2 seconds to get stable CPU measurements
        while time.time() - start_time < 2.0:
            _ = model(image)
            iterations += 1

        # Measure CPU again
        end_cpu = psutil.cpu_percent(percpu=True)
        end_cpu_avg = np.mean(end_cpu)

        # Calculate average inference time
        total_time = time.time() - start_time
        avg_inference_time = (total_time / iterations) * 1000  # in milliseconds

        # Get cpu usage increase during inference
        cpu_usage = max(0.1, end_cpu_avg - baseline_cpu_avg)

        return {
            "avg_inference_time_ms": avg_inference_time,
            "cpu_usage_percent": cpu_usage,
            "iterations": iterations,
        }

    def compare_models(self, image_path):
        image = cv2.imread(image_path)

        print("Starting comprehensive model comparison...")

        # First measure inference time and model size without CPU concerns
        print("\nMeasuring base inference speed and model size...")
        v8_base_perf = self.benchmark_inference(self.yolo8_base, image)
        v9_base_perf = self.benchmark_inference(self.yolo9_base, image)
        v8_best_perf = self.benchmark_inference(self.yolo8_best, image)
        v9_best_perf = self.benchmark_inference(self.yolo9_best, image)

        # Now measure CPU usage with a more direct approach
        print("\nMeasuring CPU usage patterns...")
        v8_base_cpu = self.benchmark_model_with_cpu(self.yolo8_base, image)
        v9_base_cpu = self.benchmark_model_with_cpu(self.yolo9_base, image)
        v8_best_cpu = self.benchmark_model_with_cpu(self.yolo8_best, image)
        v9_best_cpu = self.benchmark_model_with_cpu(self.yolo9_best, image)

        # Prepare consolidated metrics
        v8_base_metrics = {
            "inference_time_ms": v8_base_perf["avg_inference_time_ms"],
            "cpu_usage": v8_base_cpu["cpu_usage_percent"],
            "model_size": v8_base_perf["model_size_mb"],
        }

        v9_base_metrics = {
            "inference_time_ms": v9_base_perf["avg_inference_time_ms"],
            "cpu_usage": v9_base_cpu["cpu_usage_percent"],
            "model_size": v9_base_perf["model_size_mb"],
        }

        v8_best_metrics = {
            "inference_time_ms": v8_best_perf["avg_inference_time_ms"],
            "cpu_usage": v8_best_cpu["cpu_usage_percent"],
            "model_size": v8_best_perf["model_size_mb"],
        }

        v9_best_metrics = {
            "inference_time_ms": v9_best_perf["avg_inference_time_ms"],
            "cpu_usage": v9_best_cpu["cpu_usage_percent"],
            "model_size": v9_best_perf["model_size_mb"],
        }

        # Run model inference to get visualization results
        print("\nGenerating visualization results...")
        v8_base_results = self.yolo8_base(image)
        v9_base_results = self.yolo9_base(image)
        v8_best_results = self.yolo8_best(image)
        v9_best_results = self.yolo9_best(image)

        # Print comparison results
        print("\n--- Base Models Comparison ---")
        print(f"{'Metric':<20} {'YOLOv8n Base':<15} {'YOLOv9c Base':<15}")
        print("-" * 50)
        print(
            f"{'Inference Time (ms)':<20} {v8_base_metrics['inference_time_ms']:<15.2f} {v9_base_metrics['inference_time_ms']:<15.2f}"
        )
        print(
            f"{'CPU Usage (%)':<20} {v8_base_metrics['cpu_usage']:<15.2f} {v9_base_metrics['cpu_usage']:<15.2f}"
        )
        print(
            f"{'Model Size (MB)':<20} {v8_base_metrics['model_size']:<15.2f} {v9_base_metrics['model_size']:<15.2f}"
        )

        print("\n--- Best.pt Models Comparison ---")
        print(f"{'Metric':<20} {'YOLOv8 Best':<15} {'YOLOv9 Best':<15}")
        print("-" * 50)
        print(
            f"{'Inference Time (ms)':<20} {v8_best_metrics['inference_time_ms']:<15.2f} {v9_best_metrics['inference_time_ms']:<15.2f}"
        )
        print(
            f"{'CPU Usage (%)':<20} {v8_best_metrics['cpu_usage']:<15.2f} {v9_best_metrics['cpu_usage']:<15.2f}"
        )
        print(
            f"{'Model Size (MB)':<20} {v8_best_metrics['model_size']:<15.2f} {v9_best_metrics['model_size']:<15.2f}"
        )

        # Visualize detection results
        self.visualize_detections(
            [
                v8_base_results[0],
                v9_base_results[0],
                v8_best_results[0],
                v9_best_results[0],
            ],
            ["YOLOv8x Base", "YOLOv9c Base", "YOLOv8x Best", "YOLOv9 Best"],
        )

        # Visualize performance metrics
        self.visualize_performance_metrics(
            [v8_base_metrics, v9_base_metrics, v8_best_metrics, v9_best_metrics],
            ["YOLOv8x Base", "YOLOv9c Base", "YOLOv8x Best", "YOLOv9 Best"],
        )

    def visualize_detections(self, results, titles):
        """Visualize detection results in a 2x2 grid"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, (result, title) in enumerate(zip(results, titles)):
            axes[i].imshow(result.plot())
            axes[i].set_title(title)
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig("detection_comparison.png")
        plt.show()

    def visualize_performance_metrics(self, metrics_list, labels):
        """Visualize performance metrics in side-by-side bar charts"""
        # Extract metrics
        inference_times = [m["inference_time_ms"] for m in metrics_list]
        cpu_usages = [m["cpu_usage"] for m in metrics_list]
        model_sizes = [m["model_size"] for m in metrics_list]

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot inference time
        bars1 = ax1.bar(
            labels, inference_times, color=["blue", "orange", "green", "red"]
        )
        ax1.set_title("Inference Time Comparison")
        ax1.set_ylabel("Time (milliseconds)")
        ax1.set_ylim(bottom=0)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}ms",
                ha="center",
                va="bottom",
            )

        # Plot CPU usage
        bars2 = ax2.bar(labels, cpu_usages, color=["blue", "orange", "green", "red"])
        ax2.set_title("CPU Usage Comparison")
        ax2.set_ylabel("CPU Usage (%)")
        ax2.set_ylim(bottom=0)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

        # Plot model size
        bars3 = ax3.bar(labels, model_sizes, color=["blue", "orange", "green", "red"])
        ax3.set_title("Model Size Comparison")
        ax3.set_ylabel("Model Size (MB)")
        ax3.set_ylim(bottom=0)
        for bar in bars3:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}MB",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig("performance_comparison.png")
        plt.show()


if __name__ == "__main__":
    comparison = ModelComparison()
    comparison.compare_models("test.jpg")
