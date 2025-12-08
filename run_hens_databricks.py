# Databricks notebook source
# MAGIC %md
# MAGIC # HENS (Huge Ensembles) Inference - Databricks Benchmark
# MAGIC
# MAGIC This notebook runs NVIDIA Earth2Studio HENS inference for performance comparison with EC2.
# MAGIC
# MAGIC **Benchmark Configuration:** NSTEPS = [4, 40, 400]
# MAGIC
# MAGIC **Important:** The pip install cell below is NOT counted in the benchmark timing.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies (NOT TIMED)
# MAGIC Based on working HENS_Ensemble_Inference notebook

# COMMAND ----------

# Step 1: Upgrade build tools
%pip install --upgrade pip wheel hatchling
%pip install packaging==25.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Step 2: Install Earth2Studio and dependencies
%pip install --no-cache-dir --no-build-isolation "earth2studio @ git+https://github.com/NVIDIA/earth2studio"
%pip install --no-cache-dir "makani @ git+https://github.com/NVIDIA/modulus-makani.git@49280812513d8f1daf872a2e9343855a6adb3acf"
%pip install numcodecs==0.14.0
%pip install nvidia-physicsnemo
%pip install h5py==3.13.0
%pip install netCDF4==1.6.5
%pip install omegaconf==2.3.0
%pip install hydra-core==1.3.2
%pip install geopandas
%pip install boto3
%pip install esgf-pyclient
%pip install termcolor
%pip install eccodes==2.39.0
%pip install xarray==2025.1.2
%pip install zarr==3.1.3
%pip install numpy==1.26.4
%pip install --upgrade s3fs aiobotocore botocore
%pip install ruamel.yaml
%pip install moviepy
%pip install --upgrade more-itertools

# COMMAND ----------

# MAGIC %pip install cartopy

# COMMAND ----------

# Restart Python to pick up all packages
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Benchmark Starts Here
# MAGIC Everything below this point is timed for apples-to-apples comparison with EC2.

# COMMAND ----------

"""
HENS (Huge Ensembles) Inference Script for Databricks

Benchmark script for performance comparison with EC2.
Based on: NVIDIA Earth2Studio HENS example notebook
"""

import gc
import json
import os
import time
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = "/tmp/hens_outputs"
NSTEPS_LIST = [4, 40, 400]  # Multiple NSTEPS for comprehensive benchmark
NENSEMBLE = 2               # Ensemble members per checkpoint
START_DATE = "2024-01-01"

# COMMAND ----------

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

# COMMAND ----------

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
        "platform": "Databricks",
        "runtime": "DBR 16.4 ML",
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Benchmark

# COMMAND ----------

# Parse configuration
start_date = datetime.strptime(START_DATE, "%Y-%m-%d")

print("="*60)
print("HENS (Huge Ensembles) Inference - Databricks Benchmark")
print("="*60)
print(f"Output directory: {OUTPUT_DIR}")
print(f"NSTEPS values: {NSTEPS_LIST}")
print(f"Ensemble members: {NENSEMBLE} per checkpoint (x2 checkpoints)")
print(f"Start date: {START_DATE}")
print("="*60)

script_start = time.time()

# Check environment
check_environment()

# COMMAND ----------

# Run benchmark
results = run_benchmark(OUTPUT_DIR, NENSEMBLE, start_date)

total_time = time.time() - script_start
results["total_time"] = round(total_time, 2)

# COMMAND ----------

# Save results as JSON
results_file = os.path.join(OUTPUT_DIR, "benchmark_results.json")
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

# COMMAND ----------

# Save text report
report_file = os.path.join(OUTPUT_DIR, "timing_report.txt")
with open(report_file, "w") as f:
    f.write("HENS Databricks Benchmark - Timing Report\n")
    f.write(f"Run date: {datetime.now().isoformat()}\n")
    f.write(f"Runtime: DBR 16.4 ML, g6e.4xlarge\n\n")
    f.write(f"{'NSTEPS':<10} {'Model 0 (s)':<15} {'Model 1 (s)':<15} {'Avg/Step (s)':<15}\n")
    f.write("-"*60 + "\n")
    for b in results["benchmarks"]:
        avg_per_step = sum(b["time_per_step"]) / 2
        f.write(f"{b['nsteps']:<10} {b['inference_times'][0]:<15.2f} {b['inference_times'][1]:<15.2f} {avg_per_step:<15.2f}\n")
    f.write("-"*60 + "\n")
    f.write(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)\n")

print(f"[INFO] Report saved to: {report_file}")
print("[INFO] Done!")

# COMMAND ----------

# Copy results to workspace for easy access
import pathlib

src = pathlib.Path("/tmp/hens_outputs/timing_report.txt")

# Get current notebook's workspace path
notebook_path = f"/Workspace{dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()}"
dst = pathlib.Path(notebook_path).parent / "timing_report.txt"

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(src.read_text())

# Also copy JSON results
src_json = pathlib.Path("/tmp/hens_outputs/benchmark_results.json")
dst_json = pathlib.Path(notebook_path).parent / "benchmark_results.json"
dst_json.write_text(src_json.read_text())

print(f"Results copied to: {dst.parent}")
