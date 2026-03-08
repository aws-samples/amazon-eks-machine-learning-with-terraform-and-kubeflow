# Agentic AI Examples

## kagent - Kubernetes Native AI Agents

[kagent](https://github.com/kagent-dev/kagent) is a Kubernetes-native framework for building AI agents with tool capabilities and LLM integration.

## Enable kagent

### 1. Create kagent Configuration

```bash
cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup

# Create kagent.tfvars
cat <<EOF > kagent.tfvars
# Enable kagent
kagent_enabled               = true
kagent_database_type         = "sqlite"
kagent_enable_ui             = true
kagent_enable_bedrock_access = true
EOF
```

### 2. Apply Terraform

**For Quick Start (Basic) users:**
```bash
terraform apply -var-file="basic.tfvars" -var-file="kagent.tfvars"
```

**For Advanced Setup users:**
```bash
# Apply with your existing terraform command plus kagent.tfvars
terraform apply -var="profile=default" -var="region=us-west-2" \
  -var="cluster_name=my-eks-cluster" \
  -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' \
  -var="import_path=s3://<YOUR_S3_BUCKET>/eks-ml-platform/" \
  -var-file="kagent.tfvars"
```

## Configuration Options

- `kagent_version`: Helm chart version (default: `"0.7.11"`, pinned for stability - override to upgrade)
- `kagent_database_type`: Choose `"sqlite"` (default, single replica) or `"postgresql"` (HA, multi-replica)
- `kagent_enable_ui`: Enable web UI (default: `true`)
- `kagent_enable_istio_ingress`: Expose UI via Istio ingress (default: `false`)
- `kagent_enable_bedrock_access`: Enable IRSA for Amazon Bedrock access (default: `false`)

## Access kagent UI

```bash
# Port-forward (default)
kubectl port-forward -n kagent svc/kagent-ui 8080:8080

# Or via Terraform output
$(terraform output -raw kagent_ui_access_command)
```

## LLM Integration Options

kagent supports multiple LLM providers. You can use self-hosted models in EKS or cloud-based services.

### Option 1: Self-Hosted Models in EKS (Recommended)

Deploy LLM serving solutions within the same EKS cluster:

```yaml
# Example: Using vLLM for self-hosted models
apiVersion: kagent.dev/v1alpha2
kind: ModelConfig
metadata:
  name: llama-3-8b
  namespace: kagent
spec:
  provider: OpenAI  # vLLM provides OpenAI-compatible API
  model: meta-llama3-8b-instruct
  apiKeySecret: kagent-openai
  apiKeySecretKey: OPENAI_API_KEY
  openAI:
    baseUrl: http://vllm-service.inference.svc.cluster.local:8000/v1
```

See the [Inference Examples](../inference/README.md) for deploying vLLM, Ray Serve, or Triton in EKS.

### Option 2: OpenAI or Compatible APIs

A placeholder `kagent-openai` secret is automatically created. Update it with your OpenAI API key:

```bash
kubectl create secret generic kagent-openai \
  --from-literal=OPENAI_API_KEY=<your-openai-api-key> \
  -n kagent \
  --dry-run=client -o yaml | kubectl apply -f -
```

Then create a ModelConfig:

```yaml
apiVersion: kagent.dev/v1alpha2
kind: ModelConfig
metadata:
  name: gpt-4
  namespace: kagent
spec:
  provider: OpenAI
  model: gpt-4
  apiKeySecret: kagent-openai
  apiKeySecretKey: OPENAI_API_KEY
  openAI:
    baseUrl: https://api.openai.com/v1
```

### Option 3: Amazon Bedrock (Optional)

For AWS Bedrock integration, enable IRSA:

```bash
terraform apply \
  -var="kagent_enabled=true" \
  -var="kagent_enable_bedrock_access=true"
```

When enabled, an IAM role with Bedrock permissions is automatically created and attached to the kagent controller via IRSA.

```yaml
apiVersion: kagent.dev/v1alpha2
kind: ModelConfig
metadata:
  name: claude-sonnet
  namespace: kagent
spec:
  provider: Bedrock
  model: anthropic.claude-3-5-sonnet-20241022-v2:0
  region: us-west-2
```

**Note**: The module automatically configures `controller.serviceAccount.name=kagent-sa` and `controller.serviceAccount.create=false` in the Helm values when Bedrock access is enabled.

## High Availability

For production deployments with multiple controller replicas:

```bash
terraform apply \
  -var="kagent_enabled=true" \
  -var="kagent_database_type=postgresql" \
  -var="kagent_controller_replicas=3"
```
