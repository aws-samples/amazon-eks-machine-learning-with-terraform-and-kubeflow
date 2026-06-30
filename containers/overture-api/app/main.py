"""Typed REST surface over Overture transportation segments.

Endpoints:
    GET  /api/status                            — readiness + record count
    GET  /api/segments/{record_id}              — one segment by id
    GET  /api/neighbors/{connector_id}          — segments sharing a connector
    GET  /api/segments?bbox=...&limit=N         — segments in bbox
"""
from __future__ import annotations
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from . import db, queries

OVERTURE_RELEASE_TAG = os.environ.get("OVERTURE_RELEASE_TAG", "unknown")
OVERTURE_BBOX = os.environ.get("OVERTURE_BBOX", "")

app = FastAPI(
    title="overture-api",
    description="Typed REST over an Overture Maps transportation segment partition.",
    version="0.1.0",
)


@app.get("/api/status")
def status() -> dict[str, Any]:
    """Cheap readiness check. Confirms the parquet is mountable and DuckDB can query it."""
    try:
        count = db.segment_count()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "ready": True,
        "segment_count": count,
        "overture_release_tag": OVERTURE_RELEASE_TAG,
        "bbox": OVERTURE_BBOX,
    }


@app.get("/api/segments/{record_id}")
def get_segment(record_id: str) -> dict[str, Any]:
    """Fetch one segment by its Overture record id."""
    seg = queries.fetch_one_segment(record_id)
    if seg is None:
        raise HTTPException(status_code=404, detail=f"segment {record_id!r} not found")
    return seg


@app.get("/api/neighbors/{connector_id}")
def get_neighbors(connector_id: str) -> dict[str, Any]:
    """Fetch all segments sharing the given connector (intersection)."""
    segments = queries.fetch_neighbors(connector_id)
    return {"connector_id": connector_id, "count": len(segments), "segments": segments}


@app.get("/api/segments")
def get_segments_in_bbox(
    bbox: str = Query(..., description="Bounding box as 'minLon,minLat,maxLon,maxLat'"),
    limit: int = Query(30, ge=1, le=500, description="Maximum number of segments to return"),
) -> dict[str, Any]:
    """Fetch up to `limit` segments intersecting the bounding box."""
    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must have exactly 4 values")
        min_lon, min_lat, max_lon, max_lat = parts
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid bbox: {e}")

    segments = queries.fetch_in_bbox(min_lon, min_lat, max_lon, max_lat, limit)
    return {
        "bbox": [min_lon, min_lat, max_lon, max_lat],
        "limit": limit,
        "count": len(segments),
        "segments": segments,
    }
