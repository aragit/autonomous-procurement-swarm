# Multi-stage Dockerfile for the Autonomous Procurement Swarm.
#
# Stage 1 — builder:
#   Installs the package with the ``[ray,mesh,llm]`` extras plus uvicorn for
#   running the v2 FastAPI server and the Ray dashboard (optional).
#
# Stage 2 — runtime:
#   Lean production image.  A single container image is reused for all three
#   roles (Ray head, Ray worker, FastAPI server) selected at start-up via the
#   ``MESH_ROLE`` environment variable or the first positional argument.
#
# Roles:
#   MESH_ROLE=head    → joins as / starts a Ray cluster (head node)
#   MESH_ROLE=worker  → joins an existing Ray cluster as a worker
#   MESH_ROLE=api     → runs the v2 FastAPI server (api.v2:app)
#   MESH_ROLE=legacy  → runs the v1 FastAPI server (api.main:app)

# ───────────────────────── Stage 1 — builder ─────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies for psycopg/asyncpg native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest and install first (better layer caching).
COPY pyproject.toml ./

# Install the package with all optional extras needed by the mesh runtime.
# uvicorn is pulled in explicitly for the server role.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[ray,mesh,llm,dev]" uvicorn

# Copy the full source tree so force-includes (configs/base.yaml, etc.) resolve.
COPY . .

# ───────────────────────── Stage 2 — runtime ─────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system dependencies (no compiler toolchain needed).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed site-packages from the builder.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the source tree (needed for configs/, api/, mesh/, etc.).
COPY --from=builder /build /app

# Ray GCS / object-store ports (for head node / workers).
EXPOSE 6379 8265 10001 8000 8080

# Default to the API server role.
ENV MESH_ROLE=api
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "mesh.entrypoint"]
CMD ["--role", "auto"]
