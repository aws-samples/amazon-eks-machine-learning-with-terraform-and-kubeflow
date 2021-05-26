#!/bin/bash
#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
#Permission is hereby granted, free of charge, to any person obtaining a copy of this
#software and associated documentation files (the "Software"), to deal in the Software
#without restriction, including without limitation the rights to use, copy, modify,
#merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#permit persons to whom the Software is furnished to do so.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# set aws region
aws_region=$(aws configure get region)
[[ -z "${desktop_role_arn}" ]] && echo "desktop_role_arn variable required" && exit 1
[[ -z "${eks_cluster_name}" ]] && echo "eks_cluster_name env variable is required" && exit 1

if [[ ! -f ~/.aws/credentials ]] 
then
	aws configure
fi

# configure kubectl
echo "configure kubectl"
aws sts get-caller-identity
aws eks --region $aws_region update-kubeconfig --name $eks_cluster_name

# verify kubectl works
kubectl get svc || { echo 'kubectl configuration failed' ; exit 1; }
chmod go-rwx $HOME/.kube/config

# configure open id provider in our EKS cluster
echo "Create EKS cluster IAM OIDC provider"
eksctl utils associate-iam-oidc-provider --region $aws_region --cluster $eks_cluster_name --approve

DESKTOP_ROLE_USERNAME=$(echo ${desktop_role_arn} | awk '{split($0, a, "/"); print a[2]}')
# Create ConfigMap for aws-auth 
cat >$HOME/aws-auth.yaml <<EOL
apiVersion: v1
kind: ConfigMap
metadata:
  name: aws-auth
  namespace: kube-system
data:
  mapRoles: |
   - rolearn: ${eks_node_role_arn}
     username: system:node:{{EC2PrivateDNSName}}
     groups:
      - system:bootstrappers
      - system:nodes
   - rolearn: ${desktop_role_arn}
     username: $DESKTOP_ROLE_USERNAME
     groups:
      - system:masters
EOL

chown ubuntu:ubuntu $HOME/aws-auth.yaml

# apply aws-auth config map
kubectl apply -f $HOME/aws-auth.yaml -n kube-system

# remove credentials
rm  ~/.aws/credentials && echo "AWS Credentials Removed"