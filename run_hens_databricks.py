# Databricks notebook source
# MAGIC %md
# MAGIC # HENS (Huge Ensembles) Inference - Databricks Benchmark
# MAGIC
# MAGIC This notebook runs NVIDIA Earth2Studio HENS inference for performance comparison with EC2.
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
import os
import time
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION - Same as EC2 defaults
# ============================================================
OUTPUT_DIR = "/tmp/hens_outputs"
NSTEPS = 4          # Number of forecast steps (6h each)
NENSEMBLE = 2       # Ensemble members per checkpoint
START_DATE = "2024-01-01"
SKIP_VIZ = False

# COMMAND ----------

# Timing tracker
class Timer:
    def __init__(self):
        self.checkpoints = {}
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        self.checkpoints['script_start'] = self.start_time
        print(f"[TIMER] Script started at {datetime.now().isoformat()}")

    def checkpoint(self, name):
        now = time.time()
        self.checkpoints[name] = now
        elapsed = now - self.start_time
        print(f"[TIMER] {name}: {elapsed:.2f}s elapsed")

    def report(self):
        total = time.time() - self.start_time
        print("\n" + "="*60)
        print("TIMING REPORT")
        print("="*60)
        prev_time = self.start_time
        for name, ts in self.checkpoints.items():
            if name == 'script_start':
                continue
            delta = ts - prev_time
            print(f"  {name}: +{delta:.2f}s")
            prev_time = ts
        print("-"*60)
        print(f"  TOTAL: {total:.2f}s ({total/60:.2f} minutes)")
        print("="*60)
        return total

timer = Timer()

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

def run_inference(output_dir: str, nsteps: int, nensemble: int, start_date: datetime):
    """Run HENS ensemble inference."""

    timer.checkpoint("imports_start")

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

    timer.checkpoint("imports_complete")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up two model packages for each checkpoint
    print("\n[INFO] Loading model packages from HuggingFace...")
    model_package_1 = Package(
        "hf://datasets/maheshankur10/hens/earth2mip_prod_registry/sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed102",
        cache_options={
            "cache_storage": Package.default_cache("hens_1"),
            "same_names": True,
        },
    )

    model_package_2 = Package(
        "hf://datasets/maheshankur10/hens/earth2mip_prod_registry/sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed103",
        cache_options={
            "cache_storage": Package.default_cache("hens_2"),
            "same_names": True,
        },
    )

    timer.checkpoint("packages_loaded")

    # Create the data source
    print("[INFO] Initializing GFS data source...")
    data = GFS()

    timer.checkpoint("data_source_ready")

    # Run inference for each checkpoint
    start_date_str = start_date.strftime("%Y-%m-%d")

    for i, package in enumerate([model_package_1, model_package_2]):
        print(f"\n[INFO] Processing checkpoint {i+1}/2...")

        # Load SFNO model from package
        print(f"[INFO] Loading SFNO model from checkpoint {i+1}...")
        model = SFNO.load_model(package)

        timer.checkpoint(f"model_{i}_loaded")

        # Perturbation method
        noise_amplification = torch.zeros(model.input_coords()["variable"].shape[0])
        index_z500 = list(model.input_coords()["variable"]).index("z500")
        noise_amplification[index_z500] = 39.27  # z500 (0.35 * z500 skill)
        noise_amplification = noise_amplification.reshape(1, 1, 1, -1, 1, 1)

        seed_perturbation = CorrelatedSphericalGaussian(noise_amplitude=noise_amplification)
        perturbation = HemisphericCentredBredVector(
            model, data, seed_perturbation, noise_amplitude=noise_amplification
        )

        # IO object
        zarr_path = os.path.join(output_dir, f"hens_{i}.zarr")
        io = ZarrBackend(
            file_name=zarr_path,
            chunks={"ensemble": 1, "time": 1, "lead_time": 1},
            backend_kwargs={"overwrite": True},
        )

        print(f"[INFO] Running ensemble inference (nsteps={nsteps}, nensemble={nensemble})...")
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
        print(f"[INFO] Checkpoint {i+1} inference completed in {inference_time:.2f}s")

        print(io.root.tree())

        timer.checkpoint(f"inference_{i}_complete")

        # Clean up to free VRAM
        del model
        del perturbation
        gc.collect()
        torch.cuda.empty_cache()

    timer.checkpoint("all_inference_complete")

    return output_dir

# COMMAND ----------

def create_visualization(output_dir: str, start_date: datetime, lead_time: int = 4):
    """Create wind speed visualization from inference results."""

    print("\n[INFO] Creating visualization...")

    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    plot_date = start_date + timedelta(hours=int(6 * lead_time))

    # Load data from both zarr stores
    ds0 = xr.open_zarr(os.path.join(output_dir, "hens_0.zarr"))
    ds1 = xr.open_zarr(os.path.join(output_dir, "hens_1.zarr"))

    # Combine the datasets
    ds = xr.concat([ds0, ds1], dim="ensemble")

    # Calculate wind speed magnitude
    wind_speed = np.sqrt(ds.u10m**2 + ds.v10m**2)

    # Get mean and std of 4th timestep across ensemble
    mean_wind = wind_speed.isel(time=0, lead_time=lead_time).mean(dim="ensemble")
    std_wind = wind_speed.isel(time=0, lead_time=lead_time).std(dim="ensemble")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Plot mean
    p1 = ax1.contourf(
        mean_wind.coords["lon"],
        mean_wind.coords["lat"],
        mean_wind,
        levels=15,
        transform=ccrs.PlateCarree(),
        cmap="nipy_spectral",
    )
    ax1.coastlines()
    ax1.set_title(f'Mean Wind Speed\n{plot_date.strftime("%Y-%m-%d %H:%M UTC")}')
    fig.colorbar(p1, ax=ax1, label="m/s")

    # Plot standard deviation
    p2 = ax2.contourf(
        std_wind.coords["lon"],
        std_wind.coords["lat"],
        std_wind,
        levels=15,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
    )
    ax2.coastlines()
    ax2.set_title(
        f'Wind Speed Standard Deviation\n{plot_date.strftime("%Y-%m-%d %H:%M UTC")}'
    )
    fig.colorbar(p2, ax=ax2, label="m/s")

    plt.tight_layout()

    # Save the figure
    output_file = os.path.join(output_dir, f"hens_wind_{plot_date.strftime('%Y_%m_%d')}.jpg")
    plt.savefig(output_file)
    print(f"[INFO] Visualization saved to: {output_file}")

    # Display in notebook
    plt.show()

    timer.checkpoint("visualization_complete")

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
print(f"Forecast steps: {NSTEPS} (= {NSTEPS * 6}h)")
print(f"Ensemble members: {NENSEMBLE} per checkpoint (x2 checkpoints)")
print(f"Start date: {START_DATE}")
print("="*60)

# START TIMING
timer.start()

# Check environment
check_environment()
timer.checkpoint("environment_checked")

# COMMAND ----------

# Run inference
run_inference(OUTPUT_DIR, NSTEPS, NENSEMBLE, start_date)

# COMMAND ----------

# Create visualization
if not SKIP_VIZ:
    create_visualization(OUTPUT_DIR, start_date)

# COMMAND ----------

# Final report
total_time = timer.report()

# Save timing to file
timing_file = os.path.join(OUTPUT_DIR, "timing_report.txt")
with open(timing_file, "w") as f:
    f.write(f"HENS Databricks Benchmark - Timing Report\n")
    f.write(f"Run date: {datetime.now().isoformat()}\n")
    f.write(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)\n\n")
    f.write("Checkpoints:\n")
    prev_time = timer.start_time
    for name, ts in timer.checkpoints.items():
        if name == 'script_start':
            continue
        delta = ts - prev_time
        f.write(f"  {name}: +{delta:.2f}s\n")
        prev_time = ts

print(f"\n[INFO] Timing report saved to: {timing_file}")
print("[INFO] Done!")

# COMMAND ----------

import pathlib

src = pathlib.Path("/tmp/hens_outputs/timing_report.txt")

# Get current notebook's workspace path
notebook_path = f"/Workspace{dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()}"
dst = pathlib.Path(notebook_path).parent / "timing_report.txt"

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(src.read_text())