# overture-api

Thin FastAPI wrapper exposing three typed REST endpoints over an
Overture Maps transportation parquet via in-process DuckDB.

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /api/status` | Readiness + total segment count + release tag + bbox |
| `GET /api/segments/{record_id}` | One segment by Overture id |
| `GET /api/neighbors/{connector_id}` | All segments sharing the given connector |
| `GET /api/segments?bbox=minLon,minLat,maxLon,maxLat&limit=N` | Segments in bbox |

## Environment variables

### Serve mode (default `uvicorn app.main:app`)

| Variable | Required | Description |
|---|---|---|
| `FILTERED_PATH` | yes | Absolute path to the bbox-filtered parquet on the mounted FSx volume |
| `OVERTURE_RELEASE_TAG` | no | Release tag, surfaced in `/api/status` for debugging |
| `OVERTURE_BBOX` | no | bbox string, surfaced in `/api/status` for debugging |
| `UVICORN_HOST` | no | Default `0.0.0.0` |
| `UVICORN_PORT` | no | Default `80` |

### Filter mode (`python -m app.filter_to_bbox`)

| Variable | Required | Description |
|---|---|---|
| `S3_SOURCE_PREFIX` | yes | Overture release prefix, e.g. `s3://overturemaps-us-west-2/release` |
| `OVERTURE_RELEASE` | yes | Release tag, e.g. `2026-06-17.0` |
| `OVERTURE_THEME` | yes | Theme partition, e.g. `transportation` |
| `OVERTURE_TYPE` | yes | Type partition, e.g. `segment` |
| `OVERTURE_BBOX` | yes | Bounding box: `minLon,minLat,maxLon,maxLat` |
| `FILTERED_PATH` | yes | Where to write the filtered parquet |
| `AWS_REGION` | no | Bucket region; default `us-west-2` |

## How it's deployed

This image is consumed by the `overture-api` Helm chart in two places:

1. **The init Job** runs `python -m app.filter_to_bbox`. This reads the
   Overture transportation partition directly from S3 via DuckDB's `httpfs`
   extension, applies a bbox filter (parquet predicate pushdown means only
   the row groups whose bbox stats overlap the workshop bbox are fetched),
   and writes a single filtered parquet to FSx. Typical runtime: 30–90 s.
2. **The serve Deployment** uses the default `uvicorn app.main:app` command
   to expose the four endpoints above on port 80, reading from the filtered
   parquet on FSx.

The same image, different commands.

## Local development

```bash
docker build -t overture-api:0.1.0 .

# To run locally, mount a parquet:
docker run --rm -p 8080:80 \
  -e FILTERED_PATH=/data/segments.parquet \
  -v /path/to/your/segments.parquet:/data/segments.parquet:ro \
  overture-api:0.1.0
```

Then:

```bash
curl http://localhost:8080/api/status
curl http://localhost:8080/api/segments/way%2F12345
```
