"""One-shot DuckDB job: filter raw Overture parquet shards to a bbox.

Invoked by the init Job container. Reads from RAW_DIR (multiple shards),
writes a single filtered parquet to FILTERED_PATH.

Environment variables:
    RAW_DIR         — directory containing the raw Overture transportation parquets
    FILTERED_PATH   — absolute path for the filtered output parquet
    OVERTURE_BBOX   — comma-separated 'minLon,minLat,maxLon,maxLat'
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import duckdb


def main() -> int:
    raw_dir = os.environ["RAW_DIR"]
    filtered_path = os.environ["FILTERED_PATH"]
    bbox = [float(x) for x in os.environ["OVERTURE_BBOX"].split(",")]
    if len(bbox) != 4:
        print(f"ERROR: OVERTURE_BBOX must be 4 floats; got {bbox}", file=sys.stderr)
        return 1
    min_lon, min_lat, max_lon, max_lat = bbox

    Path(filtered_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(filtered_path).exists():
        print(f"Filtered parquet already at {filtered_path}; skipping.")
        return 0

    print("============================================")
    print("Overture API — bbox filter")
    print(f"  raw_dir       : {raw_dir}")
    print(f"  filtered_path : {filtered_path}")
    print(f"  bbox          : {bbox}")
    print("============================================")

    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")

    # bbox.xmin etc. live inside the Overture struct column.
    sql = f"""
        COPY (
            SELECT *
            FROM read_parquet('{raw_dir}/*.parquet')
            WHERE bbox.xmin >= ?
              AND bbox.xmax <= ?
              AND bbox.ymin >= ?
              AND bbox.ymax <= ?
        ) TO '{filtered_path}' (FORMAT PARQUET, COMPRESSION 'snappy')
    """
    con.execute(sql, [min_lon, max_lon, min_lat, max_lat])

    written = con.execute(f"SELECT COUNT(*) FROM read_parquet('{filtered_path}')").fetchone()[0]
    print(f"\nFiltered {written} segments to {filtered_path}")
    size = Path(filtered_path).stat().st_size / 1024 / 1024
    print(f"File size: {size:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
