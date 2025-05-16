# Serve MCP Gateway Registry (Experimental)

This example shows how to serve [MCP Gateway Registry](https://github.com/aarora79/mcp-gateway/tree/main).

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). 

See [What is in the YAML file](../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a given Helm chart.

## Launch MCP Gateway Registry Server

To launch MCP Gateway Registry server with a self-signed SSL certificate, replace `your-password` with your admin password and `your-polygon-api-key`[^1] with your [Polygon API Key](https://polygon.io/stocks) in the command below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug mcp-gateway-registry \
        charts/machine-learning/serving/generic-server \
        -f examples/agentic/mcp-gateway-registry/server.yaml \
        --set='admin_user=admin' \
        --set='admin_password=your-password' \
        --set='polygon_api_key=your-polygon-api-key' \
        -n kubeflow-user-example-com

[^1] The MCP Gateway comes with a built in example MCP server that retrieves stock information using the Polygon API.

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

## Monitor the Service

You can use the following commands to monitor the status of your deployment:

```bash
# List all pods in the namespace
kubectl get pods -n kubeflow-user-example-com

# List all services in the namespace
kubectl get services -n kubeflow-user-example-com

# View logs for a specific pod (replace with your actual pod name)
kubectl logs -n kubeflow-user-example-com mcp-gateway-registry-6f64945c45-6hnz6
```

## Access the MCP Registry & Gateway

To access the MCP Gateway Registry service locally, you can use port forwarding to connect to the service running in your Kubernetes cluster. The following commands will forward local ports to the service:

```bash
# Forward port 8080 to access the MCP Registry (7860)
kubectl port-forward -n kubeflow-user-example-com svc/mcp-gateway-registry 8080:7860

# Forward port 8081 to access the MCP Gateway over HTTP (80)
kubectl port-forward -n kubeflow-user-example-com svc/mcp-gateway-registry 8081:80

# Forward port 8082 to access the MCP Gateway over HTTPS (443)
kubectl port-forward -n kubeflow-user-example-com svc/mcp-gateway-registry 8082:443
```

After running one of these commands, you can access the service in your browser:
- MCP Registry: http://localhost:8080
- MCP Gateway over HTTP: http://localhost:8081
- MCP Gateway over HTTPS: https://localhost:8082

Note that each port-forward command runs in the foreground. You can press Ctrl+C to stop the port forwarding when you're done.
