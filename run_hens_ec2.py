#!/usr/bin/env python3
"""
HENS (Huge Ensembles) Inference Script for EC2

Standalone script converted from Databricks notebook for performance comparison.
Based on: NVIDIA Earth2Studio HENS example notebook

Usage:
    python run_hens.py [--output-dir OUTPUT_DIR] [--nsteps NSTEPS] [--nensemble NENSEMBLE]
"""

import argparse
import gc
import os
import sys
import time
from datetime import datetime, timedelta

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

    timer.checkpoint("visualization_complete")

def main():
    parser = argparse.ArgumentParser(description="Run HENS ensemble inference")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for results")
    parser.add_argument("--nsteps", type=int, default=4, help="Number of forecast steps (6h each)")
    parser.add_argument("--nensemble", type=int, default=2, help="Number of ensemble members per checkpoint")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization step")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    print("="*60)
    print("HENS (Huge Ensembles) Inference - EC2 Benchmark")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Forecast steps: {args.nsteps} (= {args.nsteps * 6}h)")
    print(f"Ensemble members: {args.nensemble} per checkpoint (x2 checkpoints)")
    print(f"Start date: {args.start_date}")
    print("="*60)

    timer.start()

    # Check environment
    check_environment()
    timer.checkpoint("environment_checked")

    # Run inference
    run_inference(args.output_dir, args.nsteps, args.nensemble, start_date)

    # Create visualization
    if not args.skip_viz:
        create_visualization(args.output_dir, start_date)

    # Final report
    total_time = timer.report()

    # Save timing to file
    timing_file = os.path.join(args.output_dir, "timing_report.txt")
    with open(timing_file, "w") as f:
        f.write(f"HENS EC2 Benchmark - Timing Report\n")
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

if __name__ == "__main__":
    main()
