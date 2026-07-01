"""DuckDB connection lifecycle for the wrapper service."""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path

import duckdb

FILTERED_PATH = os.environ.get(
    "FILTERED_PATH",
    "/fsx/home/overture-db/overture-transportation-ingolstadt/filtered/segments.parquet",
)


@lru_cache(maxsize=1)
def get_conn() -> duckdb.DuckDBPyConnection:
    """Return a single read-only DuckDB connection wired to the filtered parquet.

    The connection installs the `spatial` extension once and creates a `segments`
    view over the parquet file. Subsequent calls reuse the same connection.
    """
    p = Path(FILTERED_PATH)
    if not p.exists():
        raise FileNotFoundError(f"filtered parquet not found at {FILTERED_PATH}")

    con = duckdb.connect(database=":memory:", read_only=False)

    # Point DuckDB at the pre-installed extension cache baked into the image
    # (so we don't need $HOME writable or network access to LOAD extensions).
    ext_dir = os.environ.get("DUCKDB_EXTENSION_DIR")
    if ext_dir:
        con.execute(f"SET extension_directory='{ext_dir}';")
    con.execute("LOAD spatial;")

    con.execute(f"""
        CREATE OR REPLACE VIEW segments AS
        SELECT * FROM read_parquet('{FILTERED_PATH}');
    """)
    return con


def segment_count() -> int:
    return get_conn().execute("SELECT COUNT(*) FROM segments").fetchone()[0]
