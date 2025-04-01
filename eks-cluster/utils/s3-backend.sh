#!/bin/bash

[[ $# -ne 2 ]] && echo "usage:  $0 s3-bucket s3-prefix" && exit 1

export S3_BUCKET_NAME=$1
export S3_BUCKET_PREFIX=$2
export PATH_TO_BACKUP=terraform/state
export BUCKET_REGION=$(aws configure get region)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cat <<EOF > $DIR/../terraform/aws-eks-cluster-and-nodegroup/backend.tf
terraform {
    backend "s3" {
        bucket = "${S3_BUCKET_NAME}"
        key    = "${S3_BUCKET_PREFIX}/${PATH_TO_BACKUP}"
        region = "${BUCKET_REGION}"
    }
}
EOF
