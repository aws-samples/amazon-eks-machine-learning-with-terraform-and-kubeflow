# script to delete tf state localy and in s3
rm -f terraform.tfstate
rm -f terraform.tfstate.backup
rm -rf .terraform/
rm -f .terraform.lock.hcl
aws s3 rm s3://<bucket name>/<prefix>/terraform/state --region <region>
# aws s3 rm s3://harish-kubeflow-tf/tf/terraform/state --region ap-southeast-4
