#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/eks_creation_${TIMESTAMP}.log"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start logging
log_with_timestamp "Starting EKS cluster creation process"


# Initialize Terraform
log_with_timestamp "Initializing Terraform..."
terraform init -no-color -upgrade 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    log_with_timestamp "ERROR: Terraform init failed"
    exit 1
fi

# Apply Terraform configuration
log_with_timestamp "Applying Terraform configuration..."
terraform apply -no-color -auto-approve 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    log_with_timestamp "ERROR: Terraform apply failed"
    exit 1
fi

# Get cluster information
log_with_timestamp "Retrieving cluster information..."
terraform output -no-color 2>&1 | tee -a "$LOG_FILE"

log_with_timestamp "EKS cluster creation process completed successfully"
log_with_timestamp "Log file saved to: $LOG_FILE"