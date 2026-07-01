# overture-transportation-ingolstadt

Deploys an in-cluster Overture Maps API instance filtered to the
Ingolstadt urban core. Backs the AI agents workshop (v2).

## What this deploys

| Resource | Role |
|---|---|
| `overture-transportation-ingolstadt-init` Job | DuckDB reads the Overture transportation partition **directly from S3** via `httpfs`, applies a bbox filter with parquet predicate pushdown, writes the filtered result (~10–50 MB) to FSx. Runs in ~30–90 seconds. |
| `overture-transportation-ingolstadt` Deployment | The `overture-api` wrapper service exposing typed REST endpoints over the filtered parquet |
| `overture-transportation-ingolstadt` Service | ClusterIP, port 80, four endpoints |
| `overture-transportation-ingolstadt-endpoints` ConfigMap | `OVERTURE_API_URL` and metadata for agent pods to consume |

## Endpoints

After install, agents on the cluster can hit:

| Endpoint | Returns |
|---|---|
| `GET /api/status` | `{"ready": true, "segment_count": N, "overture_release_tag": ..., "bbox": ...}` |
| `GET /api/segments/{record_id}` | One Overture segment by id |
| `GET /api/neighbors/{connector_id}` | All segments sharing the given connector (intersection) |
| `GET /api/segments?bbox=minLon,minLat,maxLon,maxLat&limit=N` | Up to N segments intersecting the bbox |

Cluster-internal URL:

```
http://overture-transportation-ingolstadt.kubeflow-user-example-com.svc.cluster.local
```

## Install

```bash
chmod +x install.sh
./install.sh
```

Approximate timings:

| Step | Duration |
|---|---|
| Download (5 GB from S3) | ~2 min |
| Filter to bbox (DuckDB) | ~1 min |
| Wrapper deployment start | ~30 sec |
| Total | ~3–5 min |

## Uninstall

```bash
helm uninstall overture-transportation-ingolstadt -n kubeflow-user-example-com
```

This removes the Deployment, Service, ConfigMap, and Jobs. The
downloaded and filtered parquets remain on FSx (idempotency for
re-installs); delete them by hand if desired:

```bash
kubectl run cleanup --rm -it --restart=Never \
  --image=busybox:1.36 \
  -n kubeflow-user-example-com \
  -- rm -rf /fsx/home/overture-db/overture-transportation-ingolstadt
```

## License

Workshop scaffolding code is licensed per the repository root LICENSE.

The Overture Maps transportation theme data is licensed under
**ODbL** (Open Database License) with attribution to:

- © OpenStreetMap contributors
- © Overture Maps Foundation

Attribution is preserved in `lab/data/ATTRIBUTION.md` of the workshop
content repo (`building-ai-agents-with-claude-code-and-eks`).
