"""Parameterized SQL queries for the three typed endpoints.

These are the only queries the wrapper service runs against DuckDB. Each is
fixed at module load; participants and agents see only the typed REST surface.
"""
from __future__ import annotations
from typing import Any

from .db import get_conn


# Note: Overture transportation segments use `connectors` as a list of structs;
# each struct has a `connector_id` field. The `geometry` column is a WKB blob
# that DuckDB's spatial extension reads via ST_AsGeoJSON.

_SELECT_COLS = """
    id AS record_id,
    subtype,
    class,
    subclass,
    names,
    connectors,
    access_restrictions,
    level_rules,
    road_surface,
    road_flags,
    speed_limits,
    width_rules,
    prohibited_transitions,
    destinations,
    routes,
    sources,
    subclass_rules,
    ST_AsGeoJSON(geometry) AS geometry_geojson
"""


def fetch_one_segment(record_id: str) -> dict[str, Any] | None:
    """Return one segment by Overture id, or None if not found."""
    sql = f"SELECT {_SELECT_COLS} FROM segments WHERE id = ? LIMIT 1"
    row = get_conn().execute(sql, [record_id]).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def fetch_neighbors(connector_id: str) -> list[dict[str, Any]]:
    """Return all segments that include the given connector_id in their connectors."""
    sql = f"""
        SELECT {_SELECT_COLS}
        FROM segments
        WHERE EXISTS (
            SELECT 1
            FROM UNNEST(connectors) AS t(c)
            WHERE c.connector_id = ?
        )
    """
    rows = get_conn().execute(sql, [connector_id]).fetchall()
    return [_row_to_dict(r) for r in rows]


def fetch_in_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float, limit: int) -> list[dict[str, Any]]:
    """Return up to `limit` segments intersecting the bounding box."""
    sql = f"""
        SELECT {_SELECT_COLS}
        FROM segments
        WHERE ST_Intersects(
            geometry,
            ST_GeomFromText('POLYGON((' ||
                ? || ' ' || ? || ',' ||
                ? || ' ' || ? || ',' ||
                ? || ' ' || ? || ',' ||
                ? || ' ' || ? || ',' ||
                ? || ' ' || ? || '))'
            )
        )
        LIMIT ?
    """
    box = [
        min_lon, min_lat,
        max_lon, min_lat,
        max_lon, max_lat,
        min_lon, max_lat,
        min_lon, min_lat,
    ]
    rows = get_conn().execute(sql, box + [limit]).fetchall()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: tuple) -> dict[str, Any]:
    """Map DuckDB row to JSON-serializable dict matching _SELECT_COLS order."""
    cols = [
        "record_id", "subtype", "class", "subclass", "names",
        "connectors", "access_restrictions", "level_rules",
        "road_surface", "road_flags", "speed_limits", "width_rules",
        "prohibited_transitions", "destinations", "routes", "sources",
        "subclass_rules", "geometry_geojson",
    ]
    out = dict(zip(cols, row))
    # geometry_geojson is a JSON string; promote to dict so the response is one JSON.
    if out.get("geometry_geojson"):
        import json
        try:
            out["geometry"] = json.loads(out["geometry_geojson"])
        except (TypeError, ValueError):
            out["geometry"] = None
    out.pop("geometry_geojson", None)
    return out
