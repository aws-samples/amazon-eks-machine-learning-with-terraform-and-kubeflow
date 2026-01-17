terraform {
  required_version = ">= 1.5.1"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 2.7.0"
    }

    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">= 1.14.0"
    }

    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.17.0"
    }

    # AWSCC provider for SageMaker HyperPod
    # Required for awscc_sagemaker_cluster resource
    awscc = {
      source  = "hashicorp/awscc"
      version = ">= 1.0.0"
    }    
  }
}

# Add AWSCC provider configuration
provider "awscc" {
  region = var.region
  
  # Use the same profile as AWS provider if specified
  # Uncomment if using named profiles:
  # profile = var.profile
}