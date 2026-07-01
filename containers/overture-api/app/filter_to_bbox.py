"""One-shot DuckDB job: filter Overture transportation partition to a bbox.

Reads directly from the public Overture S3 bucket via DuckDB's httpfs
extension. Parquet predicate pushdown means only the row groups whose
bbox overlaps the workshop bbox are actually fetched — for a city-sized
extract of a ~66 GB partition, actual S3 read is under 100 MB and the
job completes in seconds.

Writes a single filtered parquet to FILTERED_PATH on the mounted FSx
volume; the serve Deployment reads it directly.

Environment variables:
    S3_SOURCE_PREFIX  — Overture release prefix, e.g. s3://overturemaps-us-west-2/release
    OVERTURE_RELEASE  — release tag, e.g. 2026-06-17.0
    OVERTURE_THEME    — theme partition, e.g. transportation
    OVERTURE_TYPE     — type partition, e.g. segment
    OVERTURE_BBOX     — comma-separated minLon,minLat,maxLon,maxLat
    FILTERED_PATH     — absolute path for the filtered output parquet
    AWS_REGION        — region hosting the bucket (default us-west-2)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import duckdb


def main() -> int:
    s3_prefix = os.environ["S3_SOURCE_PREFIX"]
    release = os.environ["OVERTURE_RELEASE"]
    theme = os.environ["OVERTURE_THEME"]
    otype = os.environ["OVERTURE_TYPE"]
    bbox_str = os.environ["OVERTURE_BBOX"]
    filtered_path = os.environ["FILTERED_PATH"]
    aws_region = os.environ.get("AWS_REGION", "us-west-2")

    bbox = [float(x) for x in bbox_str.split(",")]
    if len(bbox) != 4:
        print(f"ERROR: OVERTURE_BBOX must be 4 floats; got {bbox}", file=sys.stderr)
        return 1
    min_lon, min_lat, max_lon, max_lat = bbox

    Path(filtered_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(filtered_path).exists():
        print(f"Filtered parquet already at {filtered_path}; skipping.")
        return 0

    # DuckDB httpfs uses s3:// URIs; we strip the s3:// scheme and split
    # into bucket + key prefix to match the read_parquet() glob syntax.
    s3_glob = f"{s3_prefix}/{release}/theme={theme}/type={otype}/*"

    print("============================================")
    print("Overture API — bbox filter (DuckDB + httpfs, S3 pushdown)")
    print(f"  source        : {s3_glob}")
    print(f"  filtered_path : {filtered_path}")
    print(f"  bbox          : {bbox}")
    print(f"  aws_region    : {aws_region}")
    print("============================================")

    con = duckdb.connect(":memory:")

    # Point DuckDB at the pre-installed extension cache baked into the image.
    ext_dir = os.environ.get("DUCKDB_EXTENSION_DIR")
    if ext_dir:
        con.execute(f"SET extension_directory='{ext_dir}';")
    con.execute("LOAD httpfs;")
    con.execute("LOAD spatial;")

    # Anonymous S3 access (Overture bucket is Requester Pays-free / public read).
    con.execute(f"SET s3_region='{aws_region}';")
    # DuckDB's httpfs supports the AWS 'anonymous' auth mode; this avoids the
    # need for IAM credentials on this job's pod. If the pod has credentials
    # they'll be used automatically otherwise.
    try:
        con.execute("CREATE SECRET secret_overture (TYPE S3, PROVIDER config);")
    except duckdb.Error:
        # Some DuckDB versions don't support CREATE SECRET; anonymous access
        # via the default provider is fine for public buckets.
        pass

    # bbox is stored as a struct { xmin, xmax, ymin, ymax } in Overture's
    # parquet schema. Predicate pushdown on this struct's fields is the
    # magic that keeps this cheap — DuckDB reads only the row groups whose
    # bbox stats overlap our filter.
    sql = f"""
        COPY (
            SELECT *
            FROM read_parquet('{s3_glob}', hive_partitioning=1)
            WHERE bbox.xmin >= ?
              AND bbox.xmax <= ?
              AND bbox.ymin >= ?
              AND bbox.ymax <= ?
        ) TO '{filtered_path}' (FORMAT PARQUET, COMPRESSION 'snappy')
    """
    con.execute(sql, [min_lon, max_lon, min_lat, max_lat])

    written = con.execute(f"SELECT COUNT(*) FROM read_parquet('{filtered_path}')").fetchone()[0]
    size_mb = Path(filtered_path).stat().st_size / 1024 / 1024
    print(f"\nFiltered {written} segments to {filtered_path}")
    print(f"File size: {size_mb:.1f} MB")

    # Make the output world-readable so the non-root serve Deployment can read
    # it. This Job runs as root; the serve pod runs as the container's default
    # 'overture' user.
    fp = Path(filtered_path)
    fp.chmod(0o644)
    for p in list(fp.parents):
        try:
            p.chmod(0o755)
        except PermissionError:
            # Walking up will eventually hit an ancestor we don't own; stop.
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
