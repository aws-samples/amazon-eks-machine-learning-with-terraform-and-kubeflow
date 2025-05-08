# Serve MCP Gateway Registry (Experimental)

This example shows how to serve [MCP Gateway Registry](https://github.com/aarora79/mcp-gateway/tree/main).

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). 

See [What is in the YAML file](../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a given Helm chart.

## Launch MCP Gateway Registry Server

To launch MCP Gateway Registry server with a self-signed SSL certificate, replace  `your-password` with your admin password in the command below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug mcp-gateway-registry \
        charts/machine-learning/serving/generic-server \
        -f examples/agentic/mcp-gateway-registry/server.yaml \
        --set='admin_user=admin' \
        --set='admin_password=your-password' \
        -n kubeflow-user-example-com


To launch MCP Gateway Registry server with custom SSL certificates, and [Polygon API Key](https://polygon.io/stocks), replace  `your-password`,  `path-ssl-certs` and `path-ssl-private` and `your-polygon-api-key` with your values, and execute:

    helm install --debug mcp-gateway-registry \
            charts/machine-learning/serving/generic-server \
            -f examples/agentic/mcp-gateway-registry/server.yaml \
            --set='admin_user=admin' \
            --set='admin_password=your-password' \
            --set='ssl_certs=path-ssl-certs' \
            --set='ssl_private=path-ssl-private' \
            --set='polygon_api_key=your-polygon-api-key' \
            -n kubeflow-user-example-com

Recall, `s3://S3_BUCKET/ml-platform/` is imported as `/fsx`. So, for example, if you upload your certs under  `s3://S3_BUCKET/ml-platform/certs`, then you can access them under `/fsx/certs`.


## Stop Service

To stop the service:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm uninstall mcp-gateway-registry -n kubeflow-user-example-com

### Logs

Triton server logs are available in `/efs/home/mcp-gateway-registry/logs` folder. 
