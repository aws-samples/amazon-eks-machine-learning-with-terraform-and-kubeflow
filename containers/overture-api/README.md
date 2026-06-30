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

| Variable | Required | Description |
|---|---|---|
| `FILTERED_PATH` | yes | Absolute path to the bbox-filtered parquet on the mounted FSx volume |
| `OVERTURE_RELEASE_TAG` | no | Release tag, surfaced in `/api/status` for debugging |
| `OVERTURE_BBOX` | no | bbox string, surfaced in `/api/status` for debugging |
| `UVICORN_HOST` | no | Default `0.0.0.0` |
| `UVICORN_PORT` | no | Default `80` |

## How it's deployed

This image is consumed by the `overture-api` Helm chart in two places:

1. **The init Job** runs `python -m app.filter_to_bbox` to convert the raw
   parquet shards on FSx into a single bbox-filtered parquet.
2. **The serve Deployment** uses the default `uvicorn app.main:app` command
   to expose the four endpoints above on port 80.

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
