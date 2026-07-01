# =============================================================================
# overture-api wrapper service
# =============================================================================
# A small FastAPI application that exposes three typed REST endpoints over an
# in-process DuckDB connection against an Overture Maps transportation parquet
# on a mounted FSx volume.
#
# Image is reused by both the Helm chart's serve Deployment and its filter
# (init) Job. The Job invokes `python -m app.filter_to_bbox`; the Deployment
# invokes the default CMD which runs the FastAPI app via uvicorn.
# =============================================================================

FROM python:3.12-slim AS base

# DuckDB needs libgomp.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install the DuckDB extensions the app uses so the serve container
# doesn't try to download them at first request. Installing an extension
# writes to a cache dir; run this as root against a persistent global path
# (/opt/duckdb-extensions) that our non-root user can read at runtime.
ENV DUCKDB_EXTENSION_DIR=/opt/duckdb-extensions
RUN mkdir -p ${DUCKDB_EXTENSION_DIR} && python -c "import duckdb, os; c = duckdb.connect(':memory:'); c.execute(f\"SET extension_directory='{os.environ['DUCKDB_EXTENSION_DIR']}';\"); c.execute('INSTALL spatial;'); c.execute('INSTALL httpfs;'); c.close()"

COPY app /app/app

# Run as non-root. Create a home dir the user owns so any runtime lib that
# reaches for $HOME (DuckDB, urllib, etc.) has a writable place to fall back.
RUN groupadd -r overture \
 && useradd -r -g overture -m -d /home/overture overture \
 && chown -R overture:overture /app /home/overture ${DUCKDB_EXTENSION_DIR}
USER overture

# Point DuckDB at the pre-installed extension cache.
ENV HOME=/home/overture

EXPOSE 80

# Default command: serve. The init Job overrides this to run filter_to_bbox.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
