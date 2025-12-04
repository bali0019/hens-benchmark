#!/bin/bash
#
# Run HENS Benchmark on EC2 and Retrieve Results
#
# Usage: ./run_benchmark.sh
#
# Prerequisites:
#   - Run create_ec2.sh first (creates ec2_instance.env)
#   - Run setup_ec2.sh to setup the environment
#

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$PROJECT_DIR/ec2_instance.env"
RESULTS_DIR="$PROJECT_DIR/results"

# Load instance info
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Run create_ec2.sh and setup_ec2.sh first."
    exit 1
fi

source "$ENV_FILE"

echo "=============================================="
echo "HENS Benchmark - Run & Retrieve"
echo "=============================================="
echo "Instance: $INSTANCE_ID"
echo "IP: $PUBLIC_IP"
echo "=============================================="

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=60"
SSH="ssh $SSH_OPTS -i $KEY_FILE ubuntu@$PUBLIC_IP"
SCP="scp $SSH_OPTS -i $KEY_FILE"

# Check SSH connectivity
echo "[1/4] Checking SSH connectivity..."
if ! $SSH "echo 'SSH OK'" > /dev/null 2>&1; then
    echo "ERROR: Cannot connect to instance. Is it running?"
    exit 1
fi
echo "  Connected"

# Run benchmark
echo "[2/4] Running HENS benchmark (this may take 4-5 minutes)..."
echo "  Started at: $(date)"
$SSH 'source ~/hens_env/bin/activate && python run_hens.py --output-dir outputs 2>&1' | tee /tmp/hens_benchmark.log
echo "  Completed at: $(date)"

# Create results directory if needed
mkdir -p "$RESULTS_DIR"

# Retrieve results
echo "[3/4] Retrieving results..."
$SCP "ubuntu@$PUBLIC_IP:~/outputs/timing_report.txt" "$RESULTS_DIR/ec2_timing_report.txt"
echo "  Saved: $RESULTS_DIR/ec2_timing_report.txt"

# Try to get visualization if it exists
if $SSH "test -f ~/outputs/*.jpg" 2>/dev/null; then
    $SCP "ubuntu@$PUBLIC_IP:~/outputs/*.jpg" "$RESULTS_DIR/"
    echo "  Saved: visualization images"
fi

# Show results summary
echo "[4/4] Results summary:"
echo ""
cat "$RESULTS_DIR/ec2_timing_report.txt"

echo ""
echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "=============================================="
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Next steps:"
echo "  - Compare with Databricks results"
echo "  - Run ./teardown_ec2.sh when done"
echo "=============================================="
