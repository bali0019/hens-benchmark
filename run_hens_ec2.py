#!/usr/bin/env python3
"""
HENS (Huge Ensembles) Inference Script for EC2

Standalone script for performance comparison with Databricks.
Based on: NVIDIA Earth2Studio HENS example notebook

Usage:
    python run_hens_ec2.py [--output-dir OUTPUT_DIR] [--nensemble NENSEMBLE]
"""

import argparse
import gc
import json
import os
import time
from datetime import datetime, timedelta

# Benchmark configurations
NSTEPS_LIST = [4, 40, 400]

# Use ephemeral drive for large outputs (avoids disk full on root)
DEFAULT_OUTPUT_DIR = "/opt/dlami/nvme/hens_outputs"


def check_environment():
    """Verify required packages are available."""
    print("\n[INFO] Checking environment...")
    import importlib.util as u
    def ok(m):
        return u.find_spec(m) is not None

    checks = {
        "earth2studio": ok("earth2studio"),
        "makani": ok("makani"),
        "torch_harmonics": ok("torch_harmonics"),
        "torch": ok("torch"),
        "xarray": ok("xarray"),
        "cartopy": ok("cartopy"),
    }

    for pkg, status in checks.items():
        status_str = "OK" if status else "MISSING"
        print(f"  {pkg}: {status_str}")

    if not all(checks.values()):
        missing = [k for k, v in checks.items() if not v]
        raise ImportError(f"Missing required packages: {missing}")

    import torch
    print(f"\n[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA version: {torch.version.cuda}")


def run_benchmark(output_dir: str, nensemble: int, start_date: datetime):
    """Run HENS ensemble inference for multiple NSTEPS values."""

    import numpy as np
    import torch

    from earth2studio.data import GFS
    from earth2studio.io import ZarrBackend
    from earth2studio.models.auto import Package
    from earth2studio.models.px import SFNO
    from earth2studio.perturbation import (
        CorrelatedSphericalGaussian,
        HemisphericCentredBredVector,
    )
    from earth2studio.run import ensemble

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Results storage
    results = {
        "platform": "EC2",
        "instance_type": "g6e.4xlarge",
        "run_date": datetime.now().isoformat(),
        "nensemble": nensemble,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "benchmarks": []
    }

    # Set up two model packages
    print("\n[INFO] Loading model packages from HuggingFace...")
    model_packages = [
        Package(
            "hf://datasets/maheshankur10/hens/earth2mip_prod_registry/sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed102",
            cache_options={
                "cache_storage": Package.default_cache("hens_1"),
                "same_names": True,
            },
        ),
        Package(
            "hf://datasets/maheshankur10/hens/earth2mip_prod_registry/sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed103",
            cache_options={
                "cache_storage": Package.default_cache("hens_2"),
                "same_names": True,
            },
        ),
    ]

    # Create the data source
    print("[INFO] Initializing GFS data source...")
    data = GFS()

    start_date_str = start_date.strftime("%Y-%m-%d")

    # Run benchmark for each NSTEPS value
    for nsteps in NSTEPS_LIST:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: NSTEPS = {nsteps}")
        print(f"{'='*60}")

        benchmark_result = {
            "nsteps": nsteps,
            "forecast_hours": nsteps * 6,
            "inference_times": [],
            "time_per_step": []
        }

        for i, package in enumerate(model_packages):
            print(f"\n[INFO] Processing model {i} (nsteps={nsteps})...")

            # Load SFNO model
            model = SFNO.load_model(package)

            # Perturbation method
            noise_amplification = torch.zeros(model.input_coords()["variable"].shape[0])
            index_z500 = list(model.input_coords()["variable"]).index("z500")
            noise_amplification[index_z500] = 39.27
            noise_amplification = noise_amplification.reshape(1, 1, 1, -1, 1, 1)

            seed_perturbation = CorrelatedSphericalGaussian(noise_amplitude=noise_amplification)
            perturbation = HemisphericCentredBredVector(
                model, data, seed_perturbation, noise_amplitude=noise_amplification
            )

            # IO object
            zarr_path = os.path.join(output_dir, f"hens_nsteps{nsteps}_model{i}.zarr")
            io = ZarrBackend(
                file_name=zarr_path,
                chunks={"ensemble": 1, "time": 1, "lead_time": 1},
                backend_kwargs={"overwrite": True},
            )

            print(f"[INFO] Running inference (nsteps={nsteps}, nensemble={nensemble})...")
            inference_start = time.time()

            io = ensemble(
                [start_date_str],
                nsteps,
                nensemble,
                model,
                data,
                io,
                perturbation,
                batch_size=1,
                output_coords={"variable": np.array(["u10m", "v10m"])},
            )

            inference_time = time.time() - inference_start
            time_per_step = inference_time / nsteps

            benchmark_result["inference_times"].append(round(inference_time, 2))
            benchmark_result["time_per_step"].append(round(time_per_step, 2))

            print(f"[RESULT] Model {i}: {inference_time:.2f}s total, {time_per_step:.2f}s/step")

            # Clean up
            del model
            del perturbation
            gc.collect()
            torch.cuda.empty_cache()

        results["benchmarks"].append(benchmark_result)

        # Print summary for this NSTEPS
        times = benchmark_result["inference_times"]
        print(f"\n[SUMMARY] NSTEPS={nsteps}: Model 0={times[0]:.2f}s, Model 1={times[1]:.2f}s")
        print(f"          Variance: {abs(times[0] - times[1]):.2f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run HENS ensemble inference benchmark")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for results (default: ephemeral drive)")
    parser.add_argument("--nensemble", type=int, default=2, help="Number of ensemble members per checkpoint")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    print("="*60)
    print("HENS (Huge Ensembles) Inference - EC2 Benchmark")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"NSTEPS values: {NSTEPS_LIST}")
    print(f"Ensemble members: {args.nensemble} per checkpoint (x2 checkpoints)")
    print(f"Start date: {args.start_date}")
    print("="*60)

    script_start = time.time()

    # Check environment
    check_environment()

    # Run benchmark
    results = run_benchmark(args.output_dir, args.nensemble, start_date)

    total_time = time.time() - script_start
    results["total_time"] = round(total_time, 2)

    # Save results as JSON
    results_file = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to: {results_file}")

    # Print final summary
    print("\n" + "="*60)
    print("FINAL BENCHMARK RESULTS")
    print("="*60)
    print(f"{'NSTEPS':<10} {'Model 0 (s)':<15} {'Model 1 (s)':<15} {'Avg/Step (s)':<15}")
    print("-"*60)
    for b in results["benchmarks"]:
        avg_per_step = sum(b["time_per_step"]) / 2
        print(f"{b['nsteps']:<10} {b['inference_times'][0]:<15.2f} {b['inference_times'][1]:<15.2f} {avg_per_step:<15.2f}")
    print("-"*60)
    print(f"Total benchmark time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print("="*60)

    # Save text report
    report_file = os.path.join(args.output_dir, "timing_report.txt")
    with open(report_file, "w") as f:
        f.write("HENS EC2 Benchmark - Timing Report\n")
        f.write(f"Run date: {datetime.now().isoformat()}\n")
        f.write(f"Instance: g6e.4xlarge (NVIDIA L40S)\n\n")
        f.write(f"{'NSTEPS':<10} {'Model 0 (s)':<15} {'Model 1 (s)':<15} {'Avg/Step (s)':<15}\n")
        f.write("-"*60 + "\n")
        for b in results["benchmarks"]:
            avg_per_step = sum(b["time_per_step"]) / 2
            f.write(f"{b['nsteps']:<10} {b['inference_times'][0]:<15.2f} {b['inference_times'][1]:<15.2f} {avg_per_step:<15.2f}\n")
        f.write("-"*60 + "\n")
        f.write(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)\n")

    print(f"[INFO] Report saved to: {report_file}")
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
