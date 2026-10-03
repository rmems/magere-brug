# Multi-stage Dockerfile for magere-brug
#
# References:
#   - corinth-canal/Dockerfile   (CUDA multi-stage)
#   - grok-ozempic/Dockerfile     (CLI feature, test stage)

# ── Builder stage ───────────────────────────────────────────────────────────
FROM rust:latest AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY schemas/ schemas/
COPY configs/ configs/
COPY manifests/ manifests/
COPY scripts/ scripts/

# Build all workspace crates
RUN cargo build --workspace --release --locked

# ── Runtime stage ───────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# Copy binaries
COPY --from=builder /app/target/release/magere /usr/local/bin/
COPY --from=builder /app/target/release/magere-bridge /usr/local/bin/

# Copy schemas, configs, manifests for validation
COPY --from=builder /app/schemas/ /app/schemas/
COPY --from=builder /app/configs/ /app/configs/
COPY --from=builder /app/manifests/ /app/manifests/
COPY --from=builder /app/scripts/ /app/scripts/

# Python deps for validate_configs.py
RUN pip3 install --break-system-packages --no-cache-dir jsonschema

WORKDIR /app

# `configs/recipes/saaq-example.json` writes to the repo-relative
# `artifacts/saaq/...`, which resolves under the root-owned /app. Pre-create it
# owned by the runtime user so the documented
# `magere run-saaq configs/recipes/saaq-example.json` works in the image
# without an explicit --output-dir or mount.
RUN mkdir -p /app/artifacts && chown -R appuser:appuser /app/artifacts

USER appuser

ENTRYPOINT ["magere"]
