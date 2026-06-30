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

# DuckDB needs libgomp; aws-cli is not needed in this image (handled by the
# separate amazon/aws-cli image in the download job).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

# Run as non-root.
RUN groupadd -r overture && useradd -r -g overture overture \
 && chown -R overture:overture /app
USER overture

EXPOSE 80

# Default command: serve. The init Job overrides this to run filter_to_bbox.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
