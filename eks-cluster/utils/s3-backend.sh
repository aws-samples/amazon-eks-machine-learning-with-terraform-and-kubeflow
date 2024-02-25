#!/bin/bash

[[ $# -ne 1 ]] && echo "usage:  $0 s3-bucket" && exit 1

export S3_BUCKET_NAME=$1
export PATH_TO_BACKUP=amazon-eks-machine-learning-with-terraform-and-kubeflow/terraform/state
export BUCKET_REGION=$(aws configure get region)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cat <<EOF > $DIR/../terraform/aws-eks-cluster-and-nodegroup/backend.tf
terraform {
    backend "s3" {
        bucket = "${S3_BUCKET_NAME}"
        key    = "${PATH_TO_BACKUP}"
        region = "${BUCKET_REGION}"
    }
}
EOF
