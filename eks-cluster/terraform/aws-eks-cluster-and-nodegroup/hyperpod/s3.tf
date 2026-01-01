#################################################
# S3 Bucket for Lifecycle Scripts
#################################################

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "lifecycle_scripts" {
  count  = var.lifecycle_scripts_s3_bucket == "" ? 1 : 0
  
  # Must start with "sagemaker-" prefix for managed policy access
  bucket = "sagemaker-hyperpod-${var.cluster_name}-${data.aws_caller_identity.current.account_id}"

  tags = merge(var.tags, {
    Name = "HyperPod Lifecycle Scripts"
  })
}

resource "aws_s3_bucket_versioning" "lifecycle_scripts" {
  count  = var.lifecycle_scripts_s3_bucket == "" ? 1 : 0
  bucket = aws_s3_bucket.lifecycle_scripts[0].id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lifecycle_scripts" {
  count  = var.lifecycle_scripts_s3_bucket == "" ? 1 : 0
  bucket = aws_s3_bucket.lifecycle_scripts[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Upload default lifecycle script
resource "aws_s3_object" "on_create_script" {
  count  = var.lifecycle_scripts_s3_bucket == "" ? 1 : 0
  bucket = aws_s3_bucket.lifecycle_scripts[0].id
  key    = "config/on_create.sh"
  
  content = <<-EOF
#!/bin/bash
set -ex

# HyperPod EKS Node Lifecycle Script
echo "Starting HyperPod node initialization..."
echo "Timestamp: $(date)"

# Log system information
uname -a
hostname

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi
else
    echo "No NVIDIA GPU detected"
fi

# Check for Neuron (Trainium/Inferentia)
if command -v neuron-ls &> /dev/null; then
    echo "AWS Neuron devices detected:"
    neuron-ls
fi

echo "HyperPod node initialization complete!"
EOF

  content_type = "text/x-shellscript"
}

#################################################
# Local Values for S3
#################################################

locals {
  lifecycle_bucket_name = var.lifecycle_scripts_s3_bucket != "" ? var.lifecycle_scripts_s3_bucket : aws_s3_bucket.lifecycle_scripts[0].id
  lifecycle_bucket_arn  = var.lifecycle_scripts_s3_bucket != "" ? "arn:aws:s3:::${var.lifecycle_scripts_s3_bucket}" : aws_s3_bucket.lifecycle_scripts[0].arn
  lifecycle_s3_uri      = "s3://${local.lifecycle_bucket_name}/config/"
}
