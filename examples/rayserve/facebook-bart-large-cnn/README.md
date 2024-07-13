# Ray Serve Facebook BART Large CNN Model for Text Summarization

This example illustrates how to use [Ray Serve](../../../charts/machine-learning/training/rayserve/) Helm chart to serve [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn/blob/main/config.json) model for text summarization.

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). See [What is in the YAML file](../../../README.md#what-is-in-the-yaml-file) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Ray Serve. Build and push this container using following command (replace `aws-region` with your AWS Region name):

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     ./containers/ray-pytorch/build_tools/build_and_push.sh aws-region

## Launch Ray Service

To launch Ray Service,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-facebook-bart-large-cnn \
        charts/machine-learning/serving/rayserve/ \
        -f examples/rayserve/facebook-bart-large-cnn/rayservice.yaml -n kubeflow-user-example-com

## Stop Service

To stop the service:

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     helm uninstall rayserve-facebook-bart-large-cnn -n kubeflow-user-example-com