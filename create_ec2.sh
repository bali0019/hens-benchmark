#!/bin/bash
#
# EC2 Provisioning Script for HENS Benchmark
# Creates a g6e.4xlarge instance with NVIDIA L40S GPU
#
# Usage: ./create_ec2.sh [--ssh-cidr CIDR] [--name NAME] [--tags JSON] [--auto-shutdown MINUTES]
#
# Options:
#   --ssh-cidr CIDR        Comma-separated CIDR ranges for SSH access
#   --name NAME            Instance name tag (default: hens-benchmark)
#   --tags JSON            JSON object of additional tags (e.g. '{"Customer":"acme","Project":"benchmark"}')
#   --auto-shutdown MINS   Auto-terminate instance after MINS minutes (default: 60)
#
# Examples:
#   ./create_ec2.sh --ssh-cidr "203.0.113.0/24"
#   ./create_ec2.sh --ssh-cidr "10.0.0.0/8,192.168.1.0/24" --name my-instance
#   ./create_ec2.sh --tags '{"Customer":"acme-corp","Project":"hens"}' --name hens-test
#   ./create_ec2.sh  # Will prompt for CIDR or auto-detect current IP
#
# SAFETY: This script only CREATES resources, never deletes.
#

set -e

# Parse arguments
SSH_CIDRS=""
INSTANCE_NAME="hens-benchmark"
EXTRA_TAGS=""
AUTO_SHUTDOWN_MINS="60"  # Default: auto-shutdown after 60 minutes

while [[ $# -gt 0 ]]; do
    case $1 in
        --ssh-cidr)
            SSH_CIDRS="$2"
            shift 2
            ;;
        --name)
            INSTANCE_NAME="$2"
            shift 2
            ;;
        --tags)
            EXTRA_TAGS="$2"
            shift 2
            ;;
        --auto-shutdown)
            AUTO_SHUTDOWN_MINS="$2"
            shift 2
            ;;
        *)
            INSTANCE_NAME="$1"
            shift
            ;;
    esac
done

# Configuration
REGION="us-east-1"
INSTANCE_TYPE="g6e.4xlarge"
AMI_ID="ami-08c3a18fa2f155bbb"  # Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
KEY_NAME="hens-benchmark-key"
SECURITY_GROUP_NAME="hens-benchmark-sg"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# If no SSH CIDR provided, prompt user
if [ -z "$SSH_CIDRS" ]; then
    echo ""
    echo "No --ssh-cidr provided. Options:"
    echo "  1. Enter IP CIDR range(s) manually"
    echo "  2. Auto-detect current public IP"
    echo ""
    read -p "Enter CIDR(s) or press Enter to auto-detect: " USER_INPUT

    if [ -z "$USER_INPUT" ]; then
        echo "Auto-detecting public IP..."
        MY_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s icanhazip.com 2>/dev/null)
        if [ -z "$MY_IP" ]; then
            echo "ERROR: Could not auto-detect IP. Please provide --ssh-cidr manually."
            exit 1
        fi
        SSH_CIDRS="${MY_IP}/32"
        echo "Detected IP: $MY_IP"
    else
        SSH_CIDRS="$USER_INPUT"
    fi
fi

echo ""

echo "=============================================="
echo "HENS Benchmark - EC2 Provisioning"
echo "=============================================="
echo "Region: $REGION"
echo "Instance Type: $INSTANCE_TYPE"
echo "AMI: $AMI_ID"
echo "SSH CIDR(s): $SSH_CIDRS"
echo "Instance Name: $INSTANCE_NAME"
[ -n "$EXTRA_TAGS" ] && echo "Extra Tags: $EXTRA_TAGS"
echo "Auto-shutdown: ${AUTO_SHUTDOWN_MINS} minutes"
echo "=============================================="

# Check AWS credentials
echo "[1/6] Verifying AWS credentials..."
aws sts get-caller-identity --region $REGION > /dev/null 2>&1 || {
    echo "ERROR: AWS credentials not configured or expired"
    exit 1
}
echo "  Credentials OK"

# Get default VPC
echo "[2/6] Getting default VPC..."
VPC_ID=$(aws ec2 describe-vpcs --region $REGION \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)
echo "  VPC: $VPC_ID"

# Create or get key pair
echo "[3/6] Setting up SSH key pair..."
KEY_CREATED="false"
if aws ec2 describe-key-pairs --region $REGION --key-names $KEY_NAME > /dev/null 2>&1; then
    echo "  Key pair '$KEY_NAME' already exists (will NOT delete on teardown)"
else
    echo "  Creating new key pair '$KEY_NAME'..."
    aws ec2 create-key-pair --region $REGION \
        --key-name $KEY_NAME \
        --query 'KeyMaterial' --output text > "$PROJECT_DIR/${KEY_NAME}.pem"
    chmod 400 "$PROJECT_DIR/${KEY_NAME}.pem"
    echo "  Key saved to: $PROJECT_DIR/${KEY_NAME}.pem"
    KEY_CREATED="true"
fi

# Create or get security group
echo "[4/6] Setting up security group..."
SG_CREATED="false"
SG_ID=$(aws ec2 describe-security-groups --region $REGION \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    echo "  Creating security group '$SECURITY_GROUP_NAME'..."
    SG_ID=$(aws ec2 create-security-group --region $REGION \
        --group-name $SECURITY_GROUP_NAME \
        --description "Security group for HENS benchmark EC2" \
        --vpc-id $VPC_ID \
        --query 'GroupId' --output text)
    SG_CREATED="true"

    # Add SSH rules for each CIDR
    echo "  Adding SSH ingress rules..."
    IFS=',' read -ra CIDR_ARRAY <<< "$SSH_CIDRS"
    for CIDR in "${CIDR_ARRAY[@]}"; do
        CIDR=$(echo "$CIDR" | xargs)  # Trim whitespace
        echo "    - $CIDR"
        aws ec2 authorize-security-group-ingress --region $REGION \
            --group-id $SG_ID \
            --protocol tcp --port 22 --cidr "$CIDR" > /dev/null
    done
else
    echo "  Security group '$SECURITY_GROUP_NAME' already exists: $SG_ID (will NOT delete on teardown)"
    echo "  WARNING: Existing security group may have different CIDR rules."
    echo "           Delete it manually if you need to update SSH access."
    # Get VPC ID from existing security group (may differ from default VPC)
    SG_VPC_ID=$(aws ec2 describe-security-groups --region $REGION \
        --group-ids $SG_ID \
        --query 'SecurityGroups[0].VpcId' --output text)
    if [ "$SG_VPC_ID" != "$VPC_ID" ]; then
        echo "  Note: Security group is in VPC $SG_VPC_ID (not default VPC)"
        VPC_ID=$SG_VPC_ID
    fi
fi

# Get all subnets from the VPC (using SG's VPC if it differs from default)
SUBNET_IDS=$(aws ec2 describe-subnets --region $REGION \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[*].SubnetId' --output text)
echo "  Available subnets: $SUBNET_IDS"

# Launch EC2 instance
echo "[5/6] Launching EC2 instance..."
echo "  Instance type: $INSTANCE_TYPE (NVIDIA L40S GPU)"

# Build tag specifications (Name tag always, extra tags from JSON optional)
TAG_SPEC="ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}"
if [ -n "$EXTRA_TAGS" ]; then
    # Parse JSON and add each key-value pair as a tag
    # e.g. {"Customer":"acme","Project":"hens"} -> {Key=Customer,Value=acme},{Key=Project,Value=hens}
    PARSED_TAGS=$(echo "$EXTRA_TAGS" | python3 -c "
import sys, json
tags = json.load(sys.stdin)
print(','.join(['{Key=' + k + ',Value=' + v + '}' for k, v in tags.items()]))" 2>/dev/null) || {
        echo "ERROR: Invalid JSON for --tags. Expected format: '{\"Key\":\"Value\",\"Key2\":\"Value2\"}'"
        exit 1
    }
    TAG_SPEC="${TAG_SPEC},${PARSED_TAGS}"
fi
TAG_SPEC="${TAG_SPEC}]"

# Try each subnet until one works (handles capacity issues in specific AZs)
INSTANCE_ID=""
for SUBNET_ID in $SUBNET_IDS; do
    echo "  Trying subnet: $SUBNET_ID"
    INSTANCE_ID=$(aws ec2 run-instances --region $REGION \
        --image-id $AMI_ID \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $SG_ID \
        --subnet-id $SUBNET_ID \
        --associate-public-ip-address \
        --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
        --tag-specifications "$TAG_SPEC" \
        --query 'Instances[0].InstanceId' --output text 2>/dev/null) && break
    echo "    Capacity unavailable in this AZ, trying next..."
done

if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: Could not launch instance in any availability zone"
    exit 1
fi

echo "  Instance ID: $INSTANCE_ID"

# Wait for instance to be running
echo "[6/6] Waiting for instance to be running..."
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID
echo "  Instance is running"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances --region $REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo ""
echo "=============================================="
echo "EC2 INSTANCE READY"
echo "=============================================="
echo "Instance ID:  $INSTANCE_ID"
echo "Public IP:    $PUBLIC_IP"
echo "Instance:     $INSTANCE_TYPE (NVIDIA L40S)"
echo "Key file:     $PROJECT_DIR/${KEY_NAME}.pem"
echo ""
echo "Connect with:"
echo "  ssh -i \"$PROJECT_DIR/${KEY_NAME}.pem\" ubuntu@$PUBLIC_IP"
echo ""
echo "=============================================="

# Save instance info to file
cat > "$PROJECT_DIR/ec2_instance.env" << EOF
INSTANCE_ID=$INSTANCE_ID
PUBLIC_IP=$PUBLIC_IP
REGION=$REGION
KEY_FILE=$PROJECT_DIR/${KEY_NAME}.pem
KEY_NAME=$KEY_NAME
KEY_CREATED=$KEY_CREATED
SSH_CIDRS=$SSH_CIDRS
SG_ID=$SG_ID
SG_CREATED=$SG_CREATED
EXTRA_TAGS='$EXTRA_TAGS'
AUTO_SHUTDOWN_MINS=$AUTO_SHUTDOWN_MINS
SSH_CMD="ssh -i \"$PROJECT_DIR/${KEY_NAME}.pem\" ubuntu@$PUBLIC_IP"
SCP_CMD="scp -i \"$PROJECT_DIR/${KEY_NAME}.pem\""
EOF

echo "Instance info saved to: $PROJECT_DIR/ec2_instance.env"
echo ""
echo "Next steps:"
echo "  1. Wait ~2 minutes for instance to fully initialize"
echo "  2. Run: ./setup_ec2.sh"
