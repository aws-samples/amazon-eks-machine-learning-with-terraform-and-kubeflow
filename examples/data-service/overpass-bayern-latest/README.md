# overpass-bayern-latest

Deploys a local [Overpass API](https://overpass-api.de/) instance loaded with
the OpenStreetMap Bayern, Germany extract dated **March 9, 2026**
(`bayern-latest.osm.pbf`, ~796 MB from [Geofabrik](https://download.geofabrik.de/europe/germany/bayern.html)).


## Prerequisites

- EKS cluster with `kubectl` and `helm` configured
- PVCs `pv-fsx` and `pv-efs` bound in `kubeflow-user-example-com`
- Outbound HTTPS access confirmed from pods with Istio sidecar disabled

## Install

```bash
bash ./install.sh
```

The install script:
1. Verifies PVCs are bound
2. Runs `helm install overpass-bayern-latest`
3. Prints monitoring and smoke-test commands

## What gets deployed

| Resource | Name |
|---|---|
| Download Job | `overpass-bayern-latest-download` |
| Init Job | `overpass-bayern-latest-init` |
| Deployment | `overpass-bayern-latest` |
| Service (ClusterIP) | `overpass-bayern-latest` |
| ConfigMap | `overpass-bayern-latest-endpoints` |

## Timing

| Phase | Duration |
|---|---|
| PBF download (~796 GB) | ~1-2 min |
| Overpass database init | ~20–30 min |
| Total before serving | ~50-60 min |


## Endpoint

```
http://overpass-bayern-latest.kubeflow-user-example-com.svc.cluster.local/api/interpreter
```

Also available as `OVERPASS_API_URL` in the `overpass-bayern-latest-endpoints` ConfigMap.

## Coverage

This instance covers all of Bayern, Germany.

## Sample query

```bash
# Count roads in Ingolstadt, Bayern, Germany city centre
kubectl run smoke-test --rm -it --restart=Never \
  --image=curlimages/curl:8.5.0 \
  --annotations="sidecar.istio.io/inject=false" \
  -n kubeflow-user-example-com \
  -- curl -s \
    "http://overpass-bayern-latest.kubeflow-user-example-com.svc.cluster.local/api/interpreter" \
    --data '[out:json][timeout:25];way["highway"](48.74,11.40,48.76,11.44);out count;'
'
```

## Uninstall

```bash
helm uninstall overpass-bayern-latest -n kubeflow-user-example-com
```

Note: uninstalling does not delete the Overpass database from FSx, or the initialized database archive from EFS.
To fully reset including the database:

# Reinstall
./install.sh
```
