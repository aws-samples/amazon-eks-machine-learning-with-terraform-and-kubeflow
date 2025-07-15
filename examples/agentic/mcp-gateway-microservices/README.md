# MCP Gateway Registry - Microservices Deployment

# Serve MCP Gateway Registry (Experimental)

This example shows how to serve [MCP Gateway Registry](https://github.com/agentic-community/mcp-gateway-registry).

A comprehensive Model Context Protocol (MCP) Gateway Registry system deployed as microservices on Kubernetes (EKS). This system provides authentication, server registry, and multiple MCP servers for various tools and capabilities.

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). 

See [What is in the YAML file](../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a given Helm chart.

## 🏗️ System Architecture

The system consists of 6 microservices with clean service naming that matches `docker-compose.yml`:

| Service | Hostname | Port | Purpose |
|---------|----------|------|---------|
| **registry** | `registry:7860` | 7860, 80, 443 | Main registry UI + nginx reverse proxy |
| **auth-server** | `auth-server:8888` | 8888 | Authentication service (OAuth + Cognito) |
| **mcpgw-server** | `mcpgw-server:8003` | 8003 | MCP Gateway server |
| **currenttime-server** | `currenttime-server:8000` | 8000 | Current time MCP server |
| **fininfo-server** | `fininfo-server:8001` | 8001 | Financial information MCP server |
| **realserverfaketools-server** | `realserverfaketools-server:8002` | 8002 | Example MCP server with fake tools |

## 🚀 Quick Start

### Prerequisites

1. **EKS cluster** with `kubeflow-user-example-com` namespace
2. **EFS storage** mounted as `pv-efs` persistent volume
3. **Helm 3** installed
4. **kubectl** configured for your cluster

### 1. Clone and Navigate

```bash
git clone 
cd amazon-eks-machine-learning-with-terraform-and-kubeflow/examples/agentic/mcp-gateway-microservices
```

### 2. Create Configuration

Create your configuration file:

```bash
./deploy.sh config
```

This will prompt you for:
- **Admin credentials** (required)
- **GitHub OAuth** (optional) 
- **AWS Cognito** (optional)
- **Polygon API key** (optional, for financial data)

### 3. Deploy All Services

```bash
./deploy.sh deploy
```

This will deploy all 6 microservices in the correct order with dependency management.

### 4. Monitor Deployment

```bash
./deploy.sh monitor
```

Or check status:

```bash
./deploy.sh status
```

### 5. Access the Registry

Once deployed, access via port-forwarding:

```bash
kubectl port-forward -n kubeflow-user-example-com svc/registry 8080:7860
```

Then open: **http://localhost:8080**

## ⚙️ Configuration

### Required Environment Variables

```bash
# Admin credentials (required)
ADMIN_USER="admin"
ADMIN_PASSWORD="your-secure-password"
SECRET_KEY="your-secret-key"
AWS_REGION="us-east-1"
```

### Optional OAuth/SSO

```bash
# GitHub OAuth (optional)
GITHUB_CLIENT_ID="your-github-client-id"
GITHUB_CLIENT_SECRET="your-github-client-secret"

# AWS Cognito (optional) 
COGNITO_CLIENT_ID="your-cognito-client-id"
COGNITO_CLIENT_SECRET="your-cognito-client-secret"
COGNITO_USER_POOL_ID="your-cognito-pool-id"
```

### Optional API Keys

```bash
# Financial data (optional)
POLYGON_API_KEY="your-polygon-api-key"

# External auth server URL (optional)
AUTH_SERVER_EXTERNAL_URL="https://your-external-auth-server"
```

## 📋 Deployment Commands

| Command | Purpose |
|---------|---------|
| `./deploy.sh config` | Show current configuration |
| `./deploy.sh deploy` | Deploy all services |
| `./deploy.sh status` | Show deployment status |
| `./deploy.sh monitor` | Live monitoring dashboard |
| `./deploy.sh clean` | Remove all services |

## 🔍 Service Details

### Registry (Main Service)
- **Purpose**: Central hub with web UI, nginx reverse proxy
- **Dependencies**: auth-server
- **Startup time**: 15-20 minutes (downloads ML models)
- **Health check**: `http://registry:7860/health`

### Auth Server
- **Purpose**: Authentication, OAuth, user management
- **Dependencies**: None
- **Startup time**: 2-3 minutes
- **Health check**: `http://auth-server:8888/health`

### MCP Servers
- **Purpose**: Individual tool servers (time, finance, etc.)
- **Dependencies**: None (standalone)
- **Startup time**: 1-2 minutes each
- **Communication**: Registry connects to them automatically

## 🌐 Access Patterns

### Production Access
- **External users**: Only through registry at `http://localhost:8080`
- **Internal services**: Communicate via service names (e.g., `registry:7860`)

### Development/Debug Access
```bash
# Individual services (debug only)
kubectl port-forward -n kubeflow-user-example-com svc/auth-server 8888:8888
kubectl port-forward -n kubeflow-user-example-com svc/mcpgw-server 8003:8003
kubectl port-forward -n kubeflow-user-example-com svc/currenttime-server 8000:8000
kubectl port-forward -n kubeflow-user-example-com svc/fininfo-server 8001:8001
kubectl port-forward -n kubeflow-user-example-com svc/realserverfaketools-server 8002:8002
```

## 🔧 Troubleshooting

### Common Issues

**1. Registry showing "0/1 Ready"**
- **Cause**: Still downloading ML models (sentence-transformers)
- **Solution**: Wait 15-20 minutes, check logs:
```bash
kubectl logs -f registry-xxx -n kubeflow-user-example-com
```

**2. Services can't communicate**
- **Cause**: Wrong service names in configuration
- **Solution**: Verify service names match docker-compose.yml:
```bash
kubectl get svc -n kubeflow-user-example-com
```

**3. EFS mount issues**
- **Cause**: EFS not properly mounted
- **Solution**: Check PVC:
```bash
kubectl get pvc -n kubeflow-user-example-com
```

**4. Memory issues**
- **Cause**: ML models require significant memory
- **Solution**: Ensure registry has at least 4Gi memory limit

### Logs and Debugging

```bash
# View logs for specific service
kubectl logs -f <service-name>-xxx -n kubeflow-user-example-com

# Check pod details
kubectl describe pod <service-name>-xxx -n kubeflow-user-example-com

# Check resource usage
kubectl top pods -n kubeflow-user-example-com
```

## 🏗️ Architecture Notes

### Service Communication
- All services use **ClusterIP** (internal only)
- Service discovery via DNS (e.g., `http://registry:7860`)
- Only registry exposed externally via port-forward

### Storage
- **EFS**: Shared storage for models, logs, configurations
- **Symlinks**: Used to map EFS paths to container paths
- **Persistence**: Data survives pod restarts

### EFS Operations
The deployment performs extensive EFS operations to enable shared state across microservices, mirroring the `docker-compose.yml` volume mounts:

#### Shared Configuration Files
```bash
# All MCP servers copy the shared authentication scopes
cp /efs/mcp-gateway/scopes.yml /app/auth_server/scopes.yml
```

#### Server Registry Files
```bash
# mcpgw-server copies all server definitions from shared storage
cp -r /efs/mcp-gateway/servers/* /app/registry/servers/
```

#### Dynamic Nginx Configuration
```bash
# Registry generates and shares nginx configuration
if [ ! -f /efs/mcp-gateway/nginx/nginx_rev_proxy.conf ]; then
  sed 's/{{EC2_PUBLIC_DNS}}/localhost/g' /app/docker/nginx_rev_proxy.conf > /efs/mcp-gateway/nginx/nginx_rev_proxy.conf
fi
cp /efs/mcp-gateway/nginx/nginx_rev_proxy.conf /etc/nginx/conf.d/nginx_rev_proxy.conf
```

#### EFS Directory Structure
Each service creates these shared directories on EFS:
```
/efs/mcp-gateway/
├── servers/        # Server definitions (shared across services)
├── models/         # ML models (sentence-transformers, etc.)
├── logs/           # Centralized logging
├── auth_server/    # Authentication configurations (scopes.yml)
├── ssl/
│   ├── certs/      # SSL certificates
│   └── private/    # SSL private keys
├── secrets/
│   └── fininfo/    # Service-specific secrets
└── nginx/          # Dynamic nginx configurations
```

#### Symlink Operations
Most services create symlinks to EFS directories rather than copying:
```bash
ln -sf /efs/mcp-gateway/servers /app/registry/servers
ln -sf /efs/mcp-gateway/models /app/registry/models
ln -sf /efs/mcp-gateway/logs /app/logs
ln -sf /efs/mcp-gateway/auth_server/scopes.yml /app/scopes.yml
```

#### Purpose of EFS Operations
- **Shared configuration**: All services access the same `scopes.yml` for authentication
- **Persistent storage**: Server registrations survive pod restarts
- **Dynamic routing**: Registry updates nginx configuration that other services use
- **Centralized logging**: All services write to shared log directory
- **SSL certificates**: Shared certificates across nginx and services
- **Service discovery**: Registry shares server definitions with mcpgw-server

### Health Checks
- **Startup probes**: Extended timeouts for ML model loading
- **Readiness probes**: Service-specific health endpoints
- **Liveness probes**: Basic service availability

### Resource Management
- **Registry**: 2-4Gi memory (ML models + nginx)
- **Auth server**: 512Mi-1Gi memory
- **MCP servers**: 256-512Mi memory each
- **CPU**: Mostly I/O bound except during ML model loading

## 📚 Additional Resources

- **Source repository**: https://github.com/agentic-community/mcp-gateway-registry
- **MCP specification**: https://modelcontextprotocol.io/
- **Docker Compose version**: See `docker-compose.yml` in root
- **Kubernetes charts**: `../../../charts/machine-learning/serving/mcp-gateway-server`

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


## 🤝 Contributing

1. Test changes with docker-compose first
2. Ensure service names match between docker-compose.yml and Kubernetes
3. Update both deployment methods when adding features
4. Test with `./deploy.sh` before submitting PRs

---

**Note**: This Kubernetes deployment mirrors the docker-compose.yml configuration with identical service names and communication patterns for consistency across deployment methods.