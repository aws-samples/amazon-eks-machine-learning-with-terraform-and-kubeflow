# EKS Upgrade Ledger: 1.33 → 1.35

**Branch:** `eks-version-bump`
**Initiated:** 2026-06-23
**Target Kubernetes:** 1.35 (EKS standard support through 2027-03-26)
**Original target:** 1.36 — see [Why not 1.36?](#why-not-136) below.

This document is the canonical reference for what was changed and why. If something breaks during testing, start here.

## Why not 1.36?

K8s 1.36 went GA on EKS in June 2026, but two controllers in our stack don't yet have GA support:

| Controller | Latest GA | Supports K8s 1.36? | Status |
|---|---|---|---|
| `cluster-autoscaler` image | v1.35.0 | ❌ no `v1.36.x` published on `registry.k8s.io/autoscaling/cluster-autoscaler` | upstream pending |
| `cert-manager` | 1.20.2 | ❌ matrix tops out at K8s 1.35 | 1.21.x only in alpha as of 2026-06-23 |

We can revisit 1.36 once cert-manager 1.21 GA and CA v1.36.x ship. Both are likely within weeks.

## In-scope categories

- EKS cluster Kubernetes version
- EKS managed addons (the four we pin and the one we add)
- Kubernetes-API-coupled controllers (Karpenter, Istio, ALB controller, cert-manager, etc.)
- Helm provider versions in `versions.tf`

## Explicitly out of scope (separate follow-up PRs)

- Local Helm chart versions in `charts/` (mpi-operator, pv-efs, kubeflow-*, etc.) — unrelated to K8s version
- ML container base images (PyTorch 24.01, Ray 2.52.1, TF 2.12.0, DeepSpeed 0.13.4, vllm-neuron 0.3.0, etc.)
- HyperPod CLI bundled chart deps (pinned upstream by SageMaker HyperPod, do not touch)
- Major-version Helm jumps deferred until after 1.35 lands cleanly:
  - AWS LB Controller chart `v3.x` (currently on `1.x` line)
  - `terraform-aws-modules/iam` v6.x (currently `1.x`)
  - `terraform-aws-modules/eks/aws` karpenter submodule v21.x (currently `20.x`)
  - kube-prometheus-stack 87.x (currently `60.x`) — has CRD migrations
  - oauth2-proxy chart 10.x (currently `6.x`)
  - cert-manager 1.21 (currently `1.13.x`) — alpha only as noted above

## Version table

Columns: **Current** = what's on master. **Target** = what this PR sets. **Latest as of 2026-06-23** = what's in upstream right now. **Notes** = anything load-bearing.

### 1. EKS control plane

| Component | Current | Target | Latest | Notes |
|---|---|---|---|---|
| Kubernetes version (default in `variables.tf`) | 1.33 | **1.35** | 1.36 | See [Why not 1.36?](#why-not-136) |
| K8s version fallback in `main.tf:325` (locals.use_k8s_version) | "1.33" | **"1.35"** | — | Used when user passes "1.1*" pre-validation |

### 2. EKS managed addons (`aws_eks_addon`)

| Addon | Current | Target | Latest for 1.35 | Notes |
|---|---|---|---|---|
| aws-ebs-csi-driver | `v1.33.0-eksbuild.1` | **`v1.62.0-eksbuild.1`** | v1.62.0-eksbuild.1 | Currently the only managed addon pinned in TF. Others (VPC CNI / kube-proxy / CoreDNS) install with EKS defaults at cluster create time. |

**Decision:** keep only the EBS CSI addon explicitly pinned, matching repo's current pattern. Adding VPC CNI / kube-proxy / CoreDNS / EFS CSI as managed addons is a larger architectural change deferred to a separate PR.

### 3. Worker node AMI types

| Nodegroup | Current | Target | Notes |
|---|---|---|---|
| system | `AL2023_x86_64_STANDARD` | **unchanged** | AMI type, not AMI ID — EKS picks the latest 1.35-compatible AL2023 AMI automatically |
| nvidia | `AL2023_x86_64_NVIDIA` | **unchanged** | same |
| neuron | `AL2023_x86_64_NEURON` | **unchanged** | same |

No AMI ID is pinned. The `AL2023_x86_64_*` references resolve at cluster create time to the latest AMI for the configured K8s version. **Mismatch risk:** none.

### 4. Kubernetes-API-coupled Helm charts

| Chart | Current | Target | Latest | K8s 1.35 supported? | Notes |
|---|---|---|---|---|---|
| AWS Load Balancer Controller | `v1.13.2` | **`v1.17.1`** | 1.17.1 (in 1.x line); 3.4.0 in newer line | ✅ | Stay in 1.x line; v3.x is a separate major refactor |
| AWS EFS CSI Driver (Helm) | `3.1.7` | **`3.4.1`** | 4.3.0 | ✅ | Stay in 3.x line; v4 deferred |
| AWS FSx CSI Driver | `1.10.0` | **`1.17.0`** | 1.17.0 | ✅ | Same major, safe |
| EKS Blueprints Addons (TF module) | `1.21.0` | **`1.23.0`** | 1.23.0 | ✅ | Same major |
| IAM role for SA EKS (TF module) | `1.1.1` | **unchanged** | 6.6.1 (major rewrite) | n/a | v6.x has breaking input/output schema changes — deferred |
| cert-manager (Helm) | `1.13.3` | **`1.20.2`** | 1.20.2 GA / 1.21.0-alpha.x | ✅ | Skipping forward several minors; 1.21 alpha not safe |
| cluster-autoscaler (chart) | `9.46.6` | **`9.58.0`** | 9.58.0 | ✅ | Chart bump + image tag override below |
| cluster-autoscaler (image tag override) | (chart default) | **`v1.35.0`** | v1.35.0 | ✅ | Must match cluster K8s minor |
| Karpenter (Helm: karpenter + karpenter-crd) | `1.6.3` (var.karpenter_version default) | **`1.13.0`** | 1.13.0 | ✅ | Karpenter 1.13 compat matrix explicitly lists K8s 1.35 |
| Karpenter (TF module) | `20.37.0` | **stay in 20.x latest patch** | 21.23.0 (major) | n/a | v21 has breaking schema; in-major patch only |
| Istio (base, istiod, cni, ingress) | `1.26.0` | **`1.30.1`** | 1.30.1 | ✅ | 1.30 compat matrix lists K8s 1.32–1.36 |
| EFA Device Plugin (eks-charts) | `v0.4.4` | **`v0.5.29`** | v0.5.29 | ✅ | Chart still lives at `aws.github.io/eks-charts`. Earlier flag retracted after re-checking index. |

### 5. K8s-adjacent Helm charts (bumped to current major's latest patch)

| Chart | Current | Target | Latest in same major |
|---|---|---|---|
| kube-prometheus-stack | `60.3.0` | **`60.5.0`** | 60.5.0 (87.1.0 in new major, deferred) |
| Kueue | `0.11.4` | **unchanged** | 0.18.1 — defer (no K8s-version coupling) |
| DCGM Exporter | `4.0.4` | **unchanged** | 4.8.2 — defer |
| OAuth2 Proxy | `6.23.1` | **unchanged** | 10.7.0 (major) — defer |
| KServe | `v0.15.1` | **unchanged** | v0.19.0 — defer |
| Airflow | `1.16.0` | **unchanged** | 1.22.0 — defer |
| ACK SageMaker Controller | `1.2.15` | **unchanged** | v1.8.0 — defer |
| MLflow | `0.17.2` | **unchanged** | 1.11.2 (major) — defer |
| kagent | `0.7.11` | **unchanged** | 0.9.9 — defer |
| kmcp | `0.1.4` | **unchanged** | 0.3.0 — defer |
| KubeRay Operator | `1.4.2` | **unchanged** | chart still 1.4.2 |
| LWS image | `v0.7.0` | **unchanged** | v0.9.0 — defer |
| mpi-operator image | `0.4.0` | **unchanged** | v0.8.0 — defer |

These are bumped only if K8s 1.35 breaks them. None do, per their compat matrices. Defer the upgrades.

### 6. Terraform/Helm provider versions (`versions.tf`)

| Provider | Current | Target | Notes |
|---|---|---|---|
| terraform | `>= 1.5.1` | **unchanged** | fine |
| hashicorp/aws | `>= 2.7.0` | **unchanged** | very loose, latest will resolve |
| hashicorp/helm | `~> 2.17.0` | **unchanged** | fine |
| gavinbunney/kubectl | `>= 1.14.0` | **unchanged** | fine |
| hashicorp/awscc | `>= 1.0.0` | **unchanged** | fine |

## Mismatches / risks to watch during testing

1. ~~**EFA Device Plugin chart**~~ — resolved. Chart is at `aws.github.io/eks-charts` exactly where TF expects it; latest `v0.5.29` flows in cleanly.

2. **Karpenter minor-skip (1.6.3 → 1.13.0)** — within the same Karpenter v1 line but 7 minor versions apart. Karpenter has had non-breaking CRD evolution; should be safe, but Karpenter docs recommend a rolling upgrade rather than skipping versions on a live cluster. **Action for testing:** verify on a fresh cluster, not via in-place upgrade.

3. **Istio minor-skip (1.26 → 1.30)** — 4 minor versions. Istio's policy is N to N+2 supported in-place; we are jumping further. **Action for testing:** install fresh, do not in-place upgrade.

4. **cert-manager minor-skip (1.13 → 1.20)** — 7 minor versions. Webhook/CRD compatibility historically maintained across minors but **CRD migrations may be required**. **Action for testing:** apply CRD bundles in order if upgrading in place; on fresh install, no concern.

5. **ALB Controller (1.13 → 1.17)** — same major, but TargetGroupBinding CRD spec adds/removes fields between minors. **Action for testing:** confirm existing TargetGroupBinding resources still reconcile.

6. **Cluster Autoscaler image tag override** — chart 9.58.0 defaults to a `v1.34.x` image. We added an explicit `image.tag: v1.35.0` to the values block in `main.tf` cluster-autoscaler helm_release so it matches the cluster's K8s minor.

7. **Submodule helm provider pin** (`istio/`, `kubeflow/`, `hyperpod/`, `kagent/`, `kmcp/`) — these submodules previously had no helm provider pin or a loose `>= 2.9`, both of which resolve to `helm@3.x` on a fresh `terraform init`. The `aws-ia/eks-blueprints-addon@1.1.1` they use is incompatible with helm v3 block syntax. Pinned all five to `~> 2.17.0` matching the root module to make `terraform validate` work on master. **Pre-existing failure on master** — would have broken anyone doing a fresh init. Side-effect benefit of this PR.

## Validation plan (what should happen before merge)

- [x] `terraform validate` passes on root module (warnings only — pre-existing `data.aws_region.name` deprecation, unrelated)
- [x] `terraform validate` passes on `istio/`
- [x] `terraform validate` passes on `kubeflow/`
- [x] `terraform validate` passes on `slurm/`
- [x] `terraform validate` passes on `mlflow/`
- [x] `terraform validate` passes on `kagent/`
- [x] `terraform validate` passes on `kmcp/`
- [x] `terraform validate` passes on `hyperpod/`
- [ ] `terraform fmt` pass
- [ ] Manual smoke test on a non-production cluster: spin up `terraform apply`, exercise the demo scenarios from `examples/agentic/aws-devops-agent-demo` against the new cluster
- [ ] Spot-check that ALB Controller, cert-manager, Istio, Karpenter all reconcile after install

## Rollback

Each file changed in this PR is independently reversible. If a single controller version proves problematic, just bump that one back. If the cluster K8s upgrade itself misbehaves, revert the `k8s_version` default and the `use_k8s_version` fallback — those two lines.

## Changelog

All changes on branch `eks-version-bump`, off `master`, 2026-06-23.

### `eks-cluster/terraform/aws-eks-cluster-and-nodegroup/variables.tf`
- `k8s_version` default: `1.33` → `1.35`
- `karpenter_version` default: `1.6.3` → `1.13.0`
- `prometheus_version` default: `60.3.0` → `60.5.0`

### `eks-cluster/terraform/aws-eks-cluster-and-nodegroup/main.tf`
- `locals.use_k8s_version` fallback: `"1.33"` → `"1.35"`
- `module.eks_blueprints_addons` version: `1.21.0` → `1.23.0`
- `aws_load_balancer_controller.chart_version`: `v1.13.2` → `1.17.1` (also dropped the leading `v` to match the chart's index format)
- `aws_efs_csi_driver.chart_version`: `3.1.7` → `3.4.1`
- `aws_fsx_csi_driver.chart_version`: `1.10.0` → `1.17.0`
- `eks_addons.aws-ebs-csi-driver.addon_version`: `v1.33.0-eksbuild.1` → `v1.62.0-eksbuild.1`
- `helm_release.cert-manager.chart_version`: `1.13.3` → `1.20.2`
- `helm_release.cluster-autoscaler.version`: `9.46.6` → `9.58.0`
- `helm_release.cluster-autoscaler.values`: added `image.tag: "v1.35.0"` (chart default tracks one K8s minor behind; must match cluster)
- `helm_release.aws-efa-k8s-device-plugin.version`: `v0.4.4` → `v0.5.29`
- `locals.istio_repo_version` (line 1272): `1.26.0` → `1.30.1`

### `eks-cluster/terraform/aws-eks-cluster-and-nodegroup/istio/main.tf`
- `locals.istio_repo_version`: `1.26.0` → `1.30.1`

### Submodule helm provider pins (unblock `terraform validate`)
- `istio/versions.tf`: added `hashicorp/helm ~> 2.17.0`
- `kubeflow/versions.tf`: added `hashicorp/helm ~> 2.17.0`
- `hyperpod/versions.tf`: added `hashicorp/helm ~> 2.17.0`
- `kagent/versions.tf`: tightened `hashicorp/helm` from `>= 2.9` to `~> 2.17.0`
- `kmcp/versions.tf`: tightened `hashicorp/helm` from `>= 2.9` to `~> 2.17.0`

### Net files changed: 9
### Net lines: ~30 (most are 1-line value edits)
