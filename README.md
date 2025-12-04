# HENS Benchmark: Databricks vs EC2 Performance Comparison

Compare NVIDIA Earth2Studio HENS (Huge Ensembles) inference performance between Databricks and AWS EC2 with NVIDIA L40S GPU.

## Benchmark Results (December 2025)

| Platform | Total Time | Notes |
|----------|------------|-------|
| **Databricks** | 219.60s (3.66 min) | DBR 16.4 ML, g6e.4xlarge, single node |
| **AWS EC2** | 260.34s (4.34 min) | g6e.4xlarge, standalone |
| **Difference** | **Databricks ~16% faster** | 40.74s savings |

### Detailed Timing Breakdown

| Stage | Databricks | EC2 | Difference |
|-------|------------|-----|------------|
| Environment Check | 2.30s | 1.16s | EC2 1.14s faster |
| Imports | 16.25s | 14.92s | EC2 1.33s faster |
| Model 0 Load | 29.51s | 67.58s | DBR 38.07s faster |
| Inference 0 | 39.70s | 58.19s | DBR 18.49s faster |
| Model 1 Load | 29.77s | 53.86s | DBR 24.09s faster |
| Inference 1 | 36.30s | 33.35s | EC2 2.95s faster |
| Visualization | 65.14s | 30.93s | EC2 34.21s faster |

## Running on EC2

### Prerequisites
- **AWS CLI** installed and configured with credentials
- Credentials need EC2 permissions (launch instances, create key pairs, security groups)
- Verify setup: `aws sts get-caller-identity`

### Step 1: Create EC2 Instance
```bash
./create_ec2.sh
```
This will:
- Auto-detect your public IP for SSH access (or prompt for CIDR)
- Create SSH key pair and security group (if needed)
- Launch g6e.4xlarge instance with NVIDIA L40S GPU
- Schedule auto-shutdown after 60 minutes (configurable)

### Step 2: Setup Environment
```bash
./setup_ec2.sh
```
This will:
- Copy benchmark script and requirements to EC2
- Install uv and create Python 3.11 virtual environment
- Install all dependencies from requirements.txt
- Verify GPU and PyTorch installation
- Schedule auto-shutdown timer

### Step 3: Run Benchmark and Retrieve Results
```bash
./run_benchmark.sh
```
This will:
- Run the benchmark on EC2 via SSH
- Retrieve timing results to `results/` directory
- Display results summary

### Step 4: Cleanup
```bash
./teardown_ec2.sh
```

## Running on Databricks

### Option 1: Using Databricks Asset Bundle (Recommended)
```bash
# Configure Databricks CLI
databricks auth login --host https://your-workspace.cloud.databricks.com

# Validate, deploy, and run
databricks bundle validate
databricks bundle deploy
databricks bundle run hens_benchmark
```

### Option 2: Manual Import
1. Import `run_hens_databricks.py` as a notebook in your Databricks workspace
2. Create a cluster with:
   - Runtime: DBR 16.4 ML or later
   - Node type: `g6e.4xlarge` (or equivalent GPU instance)
   - Single node mode
3. Run the notebook - dependencies are installed automatically via `%pip`

## Project Structure

```
hens-benchmark/
├── README.md                   # This file
├── databricks.yml              # Databricks Asset Bundle config
├── create_ec2.sh               # EC2 provisioning script
├── setup_ec2.sh                # EC2 environment setup script
├── run_benchmark.sh            # Run benchmark and retrieve results
├── teardown_ec2.sh             # EC2 cleanup script
├── run_hens_ec2.py             # Benchmark script for EC2
├── run_hens_databricks.py      # Benchmark script for Databricks
├── requirements.txt            # Python dependencies (EC2)
├── requirements-full.txt       # Full pip freeze from DBR 16.4 ML
└── results/                    # Benchmark results
    ├── ec2_timing_report.txt
    └── databricks_timing_report.txt
```

## Script Options

### create_ec2.sh
```bash
./create_ec2.sh [OPTIONS]

Options:
  --ssh-cidr CIDR        Comma-separated CIDR ranges for SSH access
  --name NAME            Instance name tag (default: hens-benchmark)
  --tags JSON            Additional tags as JSON (e.g. '{"Customer":"acme"}')
  --auto-shutdown MINS   Auto-terminate after MINS minutes (default: 60)

Examples:
  ./create_ec2.sh                                    # Auto-detect IP
  ./create_ec2.sh --ssh-cidr "203.0.113.0/24"       # Specific CIDR
  ./create_ec2.sh --tags '{"Customer":"acme","Project":"hens"}'
  ./create_ec2.sh --auto-shutdown 120               # 2-hour timeout
```

### teardown_ec2.sh
```bash
./teardown_ec2.sh [OPTIONS]

Options:
  --force    Skip confirmation prompts
  --all      Delete key pair and security group (if created by script)
```

## Instance Configuration

| Component | Value |
|-----------|-------|
| **AMI** | `ami-08c3a18fa2f155bbb` (Deep Learning Base OSS Nvidia Driver GPU AMI Ubuntu 22.04) |
| **Instance Type** | `g6e.4xlarge` |
| **GPU** | NVIDIA L40S (48GB VRAM) |
| **vCPUs** | 16 |
| **Memory** | 128 GB |
| **Storage** | 200 GB gp3 |
| **Python** | 3.11 (via uv) |
| **PyTorch** | 2.6.0+cu124 |
| **CUDA** | 12.4 |

## Benchmark Parameters

| Parameter | Value |
|-----------|-------|
| Forecast steps | 4 (6-hour intervals = 24h forecast) |
| Ensemble members | 2 per checkpoint |
| Checkpoints | 2 (SFNO models from HuggingFace) |
| Output variables | u10m, v10m (10m wind components) |
| Data source | GFS (NOAA) |

## Output Files

- `timing_report.txt` - Detailed timing breakdown
- `hens_wind_YYYY_MM_DD.jpg` - Wind forecast visualization
- `hens_0.zarr/` - Forecast data from checkpoint 1
- `hens_1.zarr/` - Forecast data from checkpoint 2

## Auto-Shutdown Feature

The EC2 instance is configured to automatically terminate after the specified timeout (default: 60 minutes) to prevent runaway costs.

- Instance uses `--instance-initiated-shutdown-behavior terminate`
- `setup_ec2.sh` schedules `sudo shutdown +60` on the instance
- To cancel: SSH to instance and run `sudo shutdown -c`
- To extend: `sudo shutdown -c && sudo shutdown +120`

## Troubleshooting

### SSH Connection Refused
- Wait 2-3 minutes after instance launch for SSH to be ready
- Ensure security group allows SSH from your IP

### CUDA Not Available
- The AMI includes NVIDIA drivers pre-installed
- Verify with: `nvidia-smi`

### AWS Credentials Expired
```bash
# Refresh credentials before running scripts
aws sts get-caller-identity
```

### Capacity Issues
The script automatically tries multiple availability zones. If all fail:
```bash
# Try a different region
REGION=us-west-2 ./create_ec2.sh
```

## Prerequisites

- AWS CLI configured with appropriate credentials
- Databricks CLI (for DAB bundle deployment)
- SSH client for connecting to EC2 instances
- Bash shell (Linux/macOS/WSL)

## License

This benchmark code is provided for testing and evaluation purposes.
