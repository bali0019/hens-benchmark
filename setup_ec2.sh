#!/bin/bash
#
# EC2 Setup Script for HENS Benchmark
# Copies files and sets up Python environment on EC2
#
# Usage: ./setup_ec2.sh
#
# Prerequisites: Run create_ec2.sh first (or set ec2_instance.env manually)
#

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$PROJECT_DIR/ec2_instance.env"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"

# Load instance info
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Run create_ec2.sh first."
    exit 1
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "ERROR: $REQUIREMENTS_FILE not found."
    exit 1
fi

source "$ENV_FILE"

# Set default for AUTO_SHUTDOWN_MINS if not in env file
AUTO_SHUTDOWN_MINS="${AUTO_SHUTDOWN_MINS:-60}"

echo "=============================================="
echo "HENS Benchmark - EC2 Setup"
echo "=============================================="
echo "Instance: $INSTANCE_ID"
echo "IP: $PUBLIC_IP"
echo "Auto-shutdown: ${AUTO_SHUTDOWN_MINS} minutes"
echo "=============================================="

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
SSH="ssh $SSH_OPTS -i $KEY_FILE ubuntu@$PUBLIC_IP"
SCP="scp $SSH_OPTS -i $KEY_FILE"

# Wait for SSH to be available
echo "[1/6] Waiting for SSH to be available..."
for i in {1..30}; do
    if $SSH "echo 'SSH OK'" > /dev/null 2>&1; then
        echo "  SSH is ready"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 10
done

# Copy files to EC2
echo "[2/6] Copying files to EC2..."
$SCP "$PROJECT_DIR/run_hens_ec2.py" ubuntu@$PUBLIC_IP:~/run_hens_ec2.py
$SCP "$REQUIREMENTS_FILE" ubuntu@$PUBLIC_IP:~/
echo "  Files copied"

# Install uv and setup Python environment
echo "[3/6] Installing uv and setting up Python 3.11..."
$SSH << 'REMOTE_SCRIPT'
set -e

# Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create virtual environment with Python 3.11
echo "Creating Python 3.11 virtual environment..."
~/.local/bin/uv venv --python 3.11 ~/hens_env

echo "uv and Python 3.11 venv ready"
REMOTE_SCRIPT
echo "  uv installed, venv created"

# Install dependencies from requirements.txt
echo "[4/6] Installing Python dependencies from requirements.txt..."
$SSH << 'REMOTE_SCRIPT'
set -e
source $HOME/.local/bin/env
source ~/hens_env/bin/activate

echo "Installing dependencies from requirements.txt..."
~/.local/bin/uv pip install --index-strategy unsafe-best-match -r ~/requirements.txt

echo "Dependencies installed"
REMOTE_SCRIPT
echo "  Dependencies installed"

# Verify GPU
echo "[5/7] Verifying GPU and environment..."
$SSH << 'REMOTE_SCRIPT'
set -e
source ~/hens_env/bin/activate

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""
echo "=== Python/PyTorch ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Earth2Studio ==="
python -c "import earth2studio; print(f'earth2studio: {earth2studio.__version__}')"
REMOTE_SCRIPT

# Setup ephemeral drive for large outputs (549GB available)
echo "[6/7] Setting up ephemeral drive for outputs..."
$SSH << 'REMOTE_SCRIPT'
# Create output directory on ephemeral NVMe drive (avoids disk full on root)
sudo mkdir -p /opt/dlami/nvme/hens_outputs
sudo chown ubuntu:ubuntu /opt/dlami/nvme/hens_outputs
echo "  Ephemeral output dir ready: /opt/dlami/nvme/hens_outputs"
df -h /opt/dlami/nvme
REMOTE_SCRIPT

# Set up auto-shutdown timer
echo "[7/7] Setting up auto-shutdown timer (${AUTO_SHUTDOWN_MINS} minutes)..."
$SSH "sudo shutdown +${AUTO_SHUTDOWN_MINS} 'Auto-shutdown: Instance will terminate to save costs'"
echo "  Auto-shutdown scheduled in ${AUTO_SHUTDOWN_MINS} minutes"
echo "  (Instance will auto-terminate on shutdown)"
echo "  To cancel: ssh to instance and run 'sudo shutdown -c'"

echo ""
echo "=============================================="
echo "EC2 SETUP COMPLETE"
echo "=============================================="
echo ""
echo "To run the benchmark:"
echo "  $SSH"
echo "  source ~/hens_env/bin/activate"
echo "  python run_hens_ec2.py"
echo ""
echo "Or run directly:"
echo "  $SSH 'source ~/hens_env/bin/activate && python run_hens_ec2.py'"
echo ""
echo "Output will be saved to: /opt/dlami/nvme/hens_outputs (549GB ephemeral drive)"
echo ""
echo "=============================================="
