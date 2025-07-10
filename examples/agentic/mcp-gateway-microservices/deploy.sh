#!/bin/bash

# MCP Gateway Microservices Deployment Script for EKS
# Usage: ./deploy.sh [config|deploy|status|clean]

set -e

NAMESPACE="kubeflow-user-example-com"
CHART_PATH="../../../charts/machine-learning/serving/mcp-gateway-server"
CONFIG_FILE="mcp-gateway-config.env"

# Load configuration from .env file
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE..."
    # Export all variables from the .env file
    set -a
    source "$CONFIG_FILE"
    set +a
    echo "Configuration loaded successfully!"
else
    echo "Error: Configuration file $CONFIG_FILE not found!"
    echo "Please create the configuration file first."
    exit 1
fi

function show_config() {
    echo "=== MCP Gateway Microservices Configuration ==="
    echo "Namespace: $NAMESPACE"
    echo "Chart Path: $CHART_PATH"
    echo "Config File: $CONFIG_FILE"
    echo ""
    echo "=== Loaded Configuration ==="
    echo "Admin User: $ADMIN_USER"
    echo "AWS Region: $AWS_REGION"
    echo "Auth Server External URL: $AUTH_SERVER_EXTERNAL_URL"
    echo "GitHub Client ID: ${GITHUB_CLIENT_ID:-(not set)}"
    echo "Cognito Client ID: ${COGNITO_CLIENT_ID:-(not set)}"
    echo "Polygon API Key: ${POLYGON_API_KEY:-(not set)}"
    echo ""
    echo "Deployment Approach: MCP Gateway Server Chart (/app directory structure)"
    echo "Base Images:"
    echo "  ✓ python:3.12 (registry + auth server)"
    echo "  ✓ python:3.12 (all MCP servers)"
    echo ""
    echo "Services will clone from: https://github.com/agentic-community/mcp-gateway-registry"
    echo "Working directory: /app (matching docker-compose.yml)"
    echo ""
    echo "To modify configuration, edit: $CONFIG_FILE"
    echo ""
}

function wait_for_service() {
    local service_name=$1
    local timeout=${2:-600}  # Default 10 minutes for dependency installation
    
    # Registry needs much longer timeout due to heavy ML package installation
    if [ "$service_name" = "registry" ]; then
        timeout=1200  # 20 minutes for registry
        echo "Using extended timeout for registry service (20 minutes)..."
    elif [ "$service_name" = "auth-server" ]; then
        timeout=600   # 10 minutes for auth server (lightweight dependencies)
        echo "Using extended timeout for auth server (10 minutes)..."
    fi
    
    echo "Waiting for $service_name to be ready..."
    
    # Skip deployment condition check - it's unreliable with long startup times
    echo "Skipping deployment condition check (containers may take time to install dependencies)..."
    
    # Wait for pods to be ready with appropriate timeout
    echo "Waiting for pods to be ready (timeout: ${timeout}s)..."
    kubectl wait --for=condition=ready pod -l app=$service_name -n $NAMESPACE --timeout=${timeout}s
    
    # Verify pod is actually running
    echo "Verifying $service_name is running..."
    local pod_count=$(kubectl get pods -l app=$service_name -n $NAMESPACE --field-selector=status.phase=Running | grep -c $service_name || echo "0")
    
    if [ "$pod_count" -eq "0" ]; then
        echo "ERROR: No running pods found for $service_name"
        kubectl get pods -l app=$service_name -n $NAMESPACE
        echo "Last 20 lines of logs:"
        kubectl logs -l app=$service_name -n $NAMESPACE --tail=20 || true
        return 1
    fi
    
    echo "$service_name is ready! ($pod_count pod(s) running)"
}

function deploy_service() {
    local service_name=$1
    local values_file=$2
    shift 2
    local extra_args=("$@")
    
    echo "Deploying $service_name with git clone approach..."
    
    # Check if already deployed
    if helm list -n $NAMESPACE | grep -q "^$service_name"; then
        echo "WARNING: $service_name already exists. Upgrading..."
        helm upgrade $service_name $CHART_PATH \
            -f $values_file \
            "${extra_args[@]}" \
            -n $NAMESPACE
    else
        helm install $service_name $CHART_PATH \
            -f $values_file \
            "${extra_args[@]}" \
            -n $NAMESPACE
    fi
    
    # Wait for deployment to start
    echo "Waiting for $service_name deployment to start..."
    sleep 15
    
    # Show current status
    echo "Current pod status for $service_name:"
    kubectl get pods -l app=$service_name -n $NAMESPACE || true
}

function deploy_all() {
    echo "=== Deploying MCP Gateway Microservices ==="
    
    # Check if namespace exists
    kubectl get namespace $NAMESPACE >/dev/null 2>&1 || {
        echo "Namespace $NAMESPACE does not exist. Please create it first."
        exit 1
    }
    
    # 1. Deploy Auth Server first (other services depend on it)
    deploy_service "auth-server" \
        "auth-server.yaml" \
        --set="secret_key=$SECRET_KEY" \
        --set="admin_user=$ADMIN_USER" \
        --set="admin_password=$ADMIN_PASSWORD" \
        --set="github_client_id=$GITHUB_CLIENT_ID" \
        --set="github_client_secret=$GITHUB_CLIENT_SECRET" \
        --set="cognito_client_id=$COGNITO_CLIENT_ID" \
        --set="cognito_client_secret=$COGNITO_CLIENT_SECRET" \
        --set="cognito_user_pool_id=$COGNITO_USER_POOL_ID" \
        --set="aws_region=$AWS_REGION"
    
    wait_for_service "auth-server"
    
    # 2. Deploy Registry (main service)
    deploy_service "registry" \
        "registry-server.yaml" \
        --set="secret_key=$SECRET_KEY" \
        --set="admin_user=$ADMIN_USER" \
        --set="admin_password=$ADMIN_PASSWORD" \
        --set="auth_server_external_url=$AUTH_SERVER_EXTERNAL_URL" \
        --set="github_client_id=$GITHUB_CLIENT_ID" \
        --set="github_client_secret=$GITHUB_CLIENT_SECRET" \
        --set="cognito_client_id=$COGNITO_CLIENT_ID" \
        --set="cognito_client_secret=$COGNITO_CLIENT_SECRET" \
        --set="cognito_user_pool_id=$COGNITO_USER_POOL_ID" \
        --set="aws_region=$AWS_REGION"
    
    wait_for_service "registry"
    
    # 3. Deploy MCP Servers (can be done in parallel)
    echo "Deploying MCP servers..."
    
    deploy_service "currenttime-server" \
        "currenttime-server.yaml" \
        --set="secret_key=$SECRET_KEY" \
        --set="aws_region=$AWS_REGION" &
    
    deploy_service "fininfo-server" \
        "fininfo-server.yaml" \
        --set="secret_key=$SECRET_KEY" \
        --set="polygon_api_key=$POLYGON_API_KEY" \
        --set="aws_region=$AWS_REGION" &
    
    deploy_service "realserverfaketools-server" \
        "realserverfaketools-server.yaml" \
        --set="secret_key=$SECRET_KEY" \
        --set="aws_region=$AWS_REGION" &
    
    # Wait for background deployments
    wait
    
    # 4. Deploy MCP Gateway (depends on registry)
    deploy_service "mcpgw-server" \
        "mcpgw-server.yaml" \
        --set="secret_key=$SECRET_KEY" \
        --set="admin_user=$ADMIN_USER" \
        --set="admin_password=$ADMIN_PASSWORD" \
        --set="aws_region=$AWS_REGION"
    
    echo ""
    echo "=== All services deployed! ==="
    show_status
}

function show_status() {
    echo "=== Service Status ==="
    kubectl get pods -n $NAMESPACE | grep -E "(auth-server|registry|currenttime-server|fininfo-server|mcpgw-server|realserverfaketools-server)" || echo "No MCP services found"
    echo ""
    echo "=== Helm Releases ==="
    helm list -n $NAMESPACE | grep -E "(auth-server|registry|currenttime-server|fininfo-server|mcpgw-server|realserverfaketools-server)" || echo "No MCP Helm releases found"
    echo ""
    echo "=== Service Endpoints ==="
    kubectl get services -n $NAMESPACE | grep -E "(auth-server|registry|currenttime-server|fininfo-server|mcpgw-server|realserverfaketools-server)" || echo "No MCP services found"
    echo ""
    echo "=== Pod Details (if any issues) ==="
    for service in auth-server registry currenttime-server fininfo-server mcpgw-server realserverfaketools-server; do
        local pods=$(kubectl get pods -l app=$service -n $NAMESPACE --no-headers 2>/dev/null | wc -l)
        if [ "$pods" -gt "0" ]; then
            echo "--- $service pods ---"
            kubectl get pods -l app=$service -n $NAMESPACE
            # Show any problematic pods
            kubectl get pods -l app=$service -n $NAMESPACE --field-selector=status.phase!=Running | grep -v "No resources found" || true
        fi
    done
    echo ""
    echo "=== External Access (PRODUCTION) ==="
    echo "Registry UI (ONLY external entry point):"
    echo "  kubectl port-forward -n $NAMESPACE svc/registry 8080:7860"
    echo "  Then access: http://localhost:8080"
    echo ""
    echo "=== Internal Services (DEBUG ONLY) ==="
    echo "Auth Server:     kubectl port-forward -n $NAMESPACE svc/auth-server 8888:8888"
    echo "MCP Gateway:     kubectl port-forward -n $NAMESPACE svc/mcpgw-server 8003:8003"
    echo "CurrentTime:     kubectl port-forward -n $NAMESPACE svc/currenttime-server 8000:8000"
    echo "FinInfo:         kubectl port-forward -n $NAMESPACE svc/fininfo-server 8001:8001"
    echo "FakeTools:       kubectl port-forward -n $NAMESPACE svc/realserverfaketools-server 8002:8002"
    echo ""
    echo "NOTE: Internal services should NOT be exposed externally in production!"
    echo "All access should go through the registry at port 8080 (7860)."
}

function monitor_deployment() {
    echo "=== Deployment Monitoring ==="
    echo "Watching all MCP service deployments..."
    
    while true; do
        clear
        echo "=== Live Deployment Status - $(date) ==="
        show_status
        echo ""
        echo "Press Ctrl+C to stop monitoring"
        sleep 10
    done
}

function clean_all() {
    echo "=== Cleaning up all MCP services ==="
    
    # Uninstall in reverse order
    helm uninstall mcpgw-server -n $NAMESPACE 2>/dev/null || echo "mcpgw-server not found"
    helm uninstall realserverfaketools-server -n $NAMESPACE 2>/dev/null || echo "realserverfaketools-server not found"
    helm uninstall fininfo-server -n $NAMESPACE 2>/dev/null || echo "fininfo-server not found"
    helm uninstall currenttime-server -n $NAMESPACE 2>/dev/null || echo "currenttime-server not found"
    helm uninstall registry -n $NAMESPACE 2>/dev/null || echo "registry not found"
    helm uninstall auth-server -n $NAMESPACE 2>/dev/null || echo "auth-server not found"
    
    echo "Cleanup complete!"
}

function config() {
    echo "=== MCP Gateway Microservices Configuration ==="
    echo
    echo "This will set up your configuration for all 6 microservices:"
    echo "- registry-server (external access)"
    echo "- auth-server (internal)"
    echo "- currenttime-server (internal)"
    echo "- fininfo-server (internal)"
    echo "- mcpgw-server (internal)"
    echo "- realserverfaketools-server (internal)"
    echo
    
    # Required configuration
    read -p "Admin username [admin]: " ADMIN_USER
    ADMIN_USER=${ADMIN_USER:-admin}
    
    read -s -p "Admin password: " ADMIN_PASSWORD
    echo
    if [ -z "$ADMIN_PASSWORD" ]; then
        echo "Error: Admin password is required"
        exit 1
    fi
    
    read -s -p "Secret key for sessions: " SECRET_KEY
    echo
    if [ -z "$SECRET_KEY" ]; then
        SECRET_KEY=$(openssl rand -base64 32)
        echo "Generated secret key: $SECRET_KEY"
    fi
    
    # Optional GitHub OAuth
    echo
    echo "Optional GitHub OAuth Configuration:"
    read -p "GitHub Client ID (optional): " GITHUB_CLIENT_ID
    if [ ! -z "$GITHUB_CLIENT_ID" ]; then
        read -s -p "GitHub Client Secret: " GITHUB_CLIENT_SECRET
        echo
    fi
    
    # Optional AWS Cognito
    echo
    echo "Optional AWS Cognito Configuration:"
    read -p "Cognito Client ID (optional): " COGNITO_CLIENT_ID
    if [ ! -z "$COGNITO_CLIENT_ID" ]; then
        read -s -p "Cognito Client Secret: " COGNITO_CLIENT_SECRET
        echo
        read -p "Cognito User Pool ID: " COGNITO_USER_POOL_ID
    fi
    
    # Financial data API key
    echo
    echo "Optional Financial Data Configuration:"
    read -p "Polygon API Key (for fininfo-server, optional): " POLYGON_API_KEY
    
    # AWS Region
    echo
    read -p "AWS Region [us-east-1]: " AWS_REGION
    AWS_REGION=${AWS_REGION:-us-east-1}
    
    # External auth server URL
    echo
    echo "External Access Configuration:"
    read -p "Auth Server External URL (for external auth redirects, optional): " AUTH_SERVER_EXTERNAL_URL
    
    # Save configuration
    cat > $CONFIG_FILE << EOF
# MCP Gateway Microservices Configuration
# Generated on $(date)

# Required settings
ADMIN_USER="$ADMIN_USER"
ADMIN_PASSWORD="$ADMIN_PASSWORD"
SECRET_KEY="$SECRET_KEY"
AWS_REGION="$AWS_REGION"

# Optional GitHub OAuth
GITHUB_CLIENT_ID="$GITHUB_CLIENT_ID"
GITHUB_CLIENT_SECRET="$GITHUB_CLIENT_SECRET"

# Optional AWS Cognito
COGNITO_CLIENT_ID="$COGNITO_CLIENT_ID"
COGNITO_CLIENT_SECRET="$COGNITO_CLIENT_SECRET"
COGNITO_USER_POOL_ID="$COGNITO_USER_POOL_ID"

# Optional Financial Data
POLYGON_API_KEY="$POLYGON_API_KEY"

# External access
AUTH_SERVER_EXTERNAL_URL="$AUTH_SERVER_EXTERNAL_URL"
EOF

    echo
    echo "Configuration saved to: $CONFIG_FILE"
    echo "You can edit this file directly if needed."
    echo
    echo "Next: Run './deploy.sh deploy' to deploy all services"
}

# Main script logic
case "${1:-deploy}" in
    "config")
        show_config
        ;;
    "deploy")
        show_config
        deploy_all
        ;;
    "status")
        show_status
        ;;
    "clean")
        clean_all
        ;;
    "monitor")
        monitor_deployment
        ;;
    *)
        echo "Usage: $0 [config|deploy|status|clean|monitor]"
        echo ""
        echo "  config  - Show current configuration"
        echo "  deploy  - Deploy all microservices (default)"
        echo "  status  - Show status of deployed services"
        echo "  clean   - Remove all deployed services"
        echo "  monitor - Monitor deployment progress"
        exit 1
        ;;
esac 