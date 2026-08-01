FROM python:3.11-slim

WORKDIR /app

# Install build deps for asyncpg/psycopg
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source first (editable install + force-includes like configs/base.yaml
# require the full project tree to be present before the build runs)
COPY . .

# Production image: base package only (mock LLM backend). The real LLM
# backends (torch/transformers) live in the optional [llm] extra and are
# only needed under dev/CI.
RUN pip install --no-cache-dir -e .
