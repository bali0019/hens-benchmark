#!/bin/bash
#
# EC2 Teardown Script for HENS Benchmark
# Cleans up AWS resources created by create_ec2.sh
#
# SAFETY: Only deletes resources that were CREATED by the script,
#         not pre-existing resources that were reused.
#
# Usage: ./teardown_ec2.sh [--force]
#
# Options:
#   --force    Skip confirmation prompts
#

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$PROJECT_DIR/ec2_instance.env"
REGION="us-east-1"
KEY_NAME="hens-benchmark-key"
SECURITY_GROUP_NAME="hens-benchmark-sg"
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "=============================================="
echo "HENS Benchmark - EC2 Teardown"
echo "=============================================="

# Check AWS credentials
echo "[1/5] Verifying AWS credentials..."
aws sts get-caller-identity --region $REGION > /dev/null 2>&1 || {
    echo "ERROR: AWS credentials not configured or expired"
    exit 1
}
echo "  Credentials OK"

# Load instance info if available
INSTANCE_ID=""
SG_ID=""
SG_CREATED="false"
KEY_CREATED="false"
CUSTOMER_TAG=""

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo ""
    echo "Found ec2_instance.env:"
    echo "  Instance ID: ${INSTANCE_ID:-not set}"
    echo "  Security Group: ${SG_ID:-not set} (created: ${SG_CREATED})"
    echo "  Key Pair: ${KEY_NAME:-not set} (created: ${KEY_CREATED})"
    [ -n "$CUSTOMER_TAG" ] && echo "  Customer Tag: $CUSTOMER_TAG"
else
    echo ""
    echo "WARNING: ec2_instance.env not found."
    echo "         Cannot determine which resources were created by this script."
    echo "         Will only terminate instances found by tag."
fi

# Get instance ID by tag if not in env file (fallback: look for Customer tag)
if [ -z "$INSTANCE_ID" ] && [ -n "$CUSTOMER_TAG" ]; then
    echo ""
    echo "Looking for instances with Customer tag '$CUSTOMER_TAG'..."
    INSTANCE_ID=$(aws ec2 describe-instances --region $REGION \
        --filters "Name=tag:Customer,Values=$CUSTOMER_TAG" "Name=instance-state-name,Values=running,stopped,pending" \
        --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "")
    if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
        echo "  Found instance: $INSTANCE_ID"
    else
        INSTANCE_ID=""
    fi
fi

# Summary of what will be deleted
echo ""
echo "=============================================="
echo "Resources to be deleted:"
echo "=============================================="
[ -n "$INSTANCE_ID" ] && echo "  - EC2 Instance: $INSTANCE_ID (always deleted)"
if [ "$SG_CREATED" = "true" ]; then
    [ -n "$SG_ID" ] && echo "  - Security Group: $SG_ID (was created by script)"
else
    [ -n "$SG_ID" ] && echo "  - Security Group: $SG_ID - SKIPPED (pre-existing)"
fi
if [ "$KEY_CREATED" = "true" ]; then
    echo "  - Key Pair: $KEY_NAME (was created by script)"
    [ -f "$PROJECT_DIR/${KEY_NAME}.pem" ] && echo "  - Local key file: ${KEY_NAME}.pem"
else
    echo "  - Key Pair: $KEY_NAME - SKIPPED (pre-existing)"
fi
[ -f "$ENV_FILE" ] && echo "  - Environment file: ec2_instance.env"
echo "=============================================="

# Confirm deletion
if [ "$FORCE" != "true" ]; then
    echo ""
    read -p "Are you sure you want to delete these resources? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi
fi

# Terminate EC2 instance (always delete - it was created by the script)
echo ""
echo "[2/5] Terminating EC2 instance..."
if [ -n "$INSTANCE_ID" ]; then
    aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID > /dev/null 2>&1 || true
    echo "  Waiting for instance to terminate..."
    aws ec2 wait instance-terminated --region $REGION --instance-ids $INSTANCE_ID 2>/dev/null || true
    echo "  Instance terminated: $INSTANCE_ID"
else
    echo "  No instance to terminate"
fi

# Delete security group only if it was CREATED by the script
echo ""
echo "[3/5] Deleting security group..."
if [ "$SG_CREATED" = "true" ] && [ -n "$SG_ID" ]; then
    # Retry a few times in case instance is still terminating
    for i in {1..5}; do
        if aws ec2 delete-security-group --region $REGION --group-id $SG_ID 2>/dev/null; then
            echo "  Security group deleted: $SG_ID"
            break
        else
            if [ $i -lt 5 ]; then
                echo "  Waiting for dependencies to clear... ($i/5)"
                sleep 10
            else
                echo "  WARNING: Could not delete security group. May need manual cleanup."
            fi
        fi
    done
else
    if [ -n "$SG_ID" ]; then
        echo "  Skipping security group $SG_ID (pre-existing, not created by script)"
    else
        echo "  No security group to delete"
    fi
fi

# Delete key pair only if it was CREATED by the script
echo ""
echo "[4/5] Deleting key pair..."
if [ "$KEY_CREATED" = "true" ]; then
    if aws ec2 describe-key-pairs --region $REGION --key-names $KEY_NAME > /dev/null 2>&1; then
        aws ec2 delete-key-pair --region $REGION --key-name $KEY_NAME
        echo "  Key pair deleted: $KEY_NAME"
    else
        echo "  Key pair not found (already deleted)"
    fi
    # Delete local key file
    if [ -f "$PROJECT_DIR/${KEY_NAME}.pem" ]; then
        rm "$PROJECT_DIR/${KEY_NAME}.pem"
        echo "  Deleted local key file: ${KEY_NAME}.pem"
    fi
else
    echo "  Skipping key pair $KEY_NAME (pre-existing, not created by script)"
fi

# Clean up env file
echo ""
echo "[5/5] Cleaning up local files..."
if [ -f "$ENV_FILE" ]; then
    rm "$ENV_FILE"
    echo "  Deleted: ec2_instance.env"
fi

echo ""
echo "=============================================="
echo "TEARDOWN COMPLETE"
echo "=============================================="
echo "Deleted resources:"
[ -n "$INSTANCE_ID" ] && echo "  - EC2 Instance: $INSTANCE_ID"
[ "$SG_CREATED" = "true" ] && [ -n "$SG_ID" ] && echo "  - Security Group: $SG_ID"
[ "$KEY_CREATED" = "true" ] && echo "  - Key Pair: $KEY_NAME"
echo ""
echo "Preserved resources (pre-existing):"
[ "$SG_CREATED" != "true" ] && [ -n "$SG_ID" ] && echo "  - Security Group: $SG_ID"
[ "$KEY_CREATED" != "true" ] && echo "  - Key Pair: $KEY_NAME"
echo ""
