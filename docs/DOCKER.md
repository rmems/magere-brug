# Docker Usage

## Quick Start

```bash
# Build the image
docker build -t magere-brug .

# Run CLI validation
docker run --rm magere-brug validate manifests/examples/deepseek-coder-v2-lite.json

# Run Python config validation
docker run --rm magere-brug python3 scripts/validate_configs.py configs/models/batch-a-gguf-routing.json

# Interactive shell
docker-compose run --rm shell
```

## Services

| Service | Purpose |
|---------|---------|
| `validate-manifests` | Run Rust CLI manifest validation |
| `validate-configs` | Run Python config validation |
| `shell` | Interactive development shell |

## Mounting Local Models

```bash
# Mount a local GGUF directory
docker run --rm -v /path/to/models:/models:ro magere-brug validate /models/my-model.json
```

## Multi-stage Build

The Dockerfile uses a multi-stage build:
- **Builder**: Compiles all workspace crates with `--release`
- **Runtime**: Minimal Debian image with Python3 for script validation

## GitHub Actions

The `.github/workflows/docker.yml` workflow:
- Builds on every push to `main`
- Pushes tagged images on version tags (`v*`)
- Uses GitHub Actions cache for Docker layers
- Publishes to `ghcr.io/rmems/magere-brug`
