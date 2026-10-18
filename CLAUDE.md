@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Project purpose, scope boundaries, key constraints, related repositories, and the
base build commands are in `AGENTS.md` (imported above); do not duplicate them here.

## Commands

Run from the repo root. "Verified" means it ran successfully on Rust 1.97.0 (stable, Linux, CPU only).
There is no `rust-toolchain.toml`; CI uses `dtolnay/rust-toolchain@stable`.

| Command | Status |
|---|---|
| `cargo fmt --check` | verified |
| `cargo clippy --workspace --all-targets -- -D warnings` | verified |
| `cargo test --workspace` | verified |
| `cargo build --release -p magere-cli` | not run (debug `cargo build -p magere-cli` verified instead) |
| `./target/release/magere validate manifests/examples/<file>.json` for every file in `manifests/examples/` | verified with the debug binary `target/debug/magere` (5 of 5) |
| `./target/release/magere recipe validate configs/recipes/<file>.json` for every top-level file in `configs/recipes/` | verified with the debug binary `target/debug/magere` (9 of 9) |
| `cd scripts && uv sync --extra dev && uv run pytest tests/ -v` | verified (17 passed) |
| `python3 scripts/validate_configs.py configs/models/*.json` | verified |
| `python3 scripts/check_required_fields.py manifests/examples/*.json` | verified |
| `python3 scripts/check_artifact_shape.py manifests/examples/*.json` | verified |
| `docker build .` (`docker.yml`) and the manual `vultr-smoke.yml` workflow | not run (Docker and cloud runner not available) |

Run fmt and clippy before committing; CI fails on either.

## Architecture map

- `crates/magere-cli`: the `magere` binary (`src/main.rs`). Manifests, registry, checksums, `pack/` (GOZ1 packing), `recipe/`, and the recipe-driven SAAQ runner in `saaq.rs`.
- `crates/magere-corinth-core`: CPU-only SAAQ validation pipeline (telemetry, projector, funnel, latent, metric).
- `crates/magere-grok-process`: Grok-1 quantization engine (artifact, quantizer, precision, weight pack). It is `GPL-3.0-or-later`, unlike the other crates.
- `crates/magere-bridge`: placeholder `main.rs` that only prints "Hello, world!".
- `schemas/` holds `model_manifest.schema.json` and `recipe.schema.json`; `manifests/examples/` and `configs/{models,recipes}/` are the validated fixtures.
- `scripts/`: stdlib Python helpers plus `tests/test_manifest.py`; `pyproject.toml` and `uv.lock` live here, not at the repo root.
- `docs/ARCHITECTURE.md` and `docs/DOCKER.md`.

## Invariants and non-goals

- `docs-check.yml` fails when both `README.md` and `README` are missing, or when `docs/ARCHITECTURE.md` or `schemas/model_manifest.schema.json` is missing.
- It also rejects absolute personal paths in `manifests/examples/*.json`, `configs/models/*.json`, `configs/models/cloud/*.json`, and `configs/recipes/*.json` (`/home/<user>/`, `/Users/<user>/`, `C:\Users\`), and `/home/<user>/` in `scripts/*.py`. Use generic paths like `/models/...`.
- Cloud stubs in `configs/models/cloud/` must hold placeholder markers, never real credentials.
- A schema change needs matching updates to `manifests/examples/` and `configs/`. `manifest-validate.yml` runs `magere validate` on the manifests and `magere recipe validate` on `configs/recipes/`, plus `scripts/validate_configs.py` (model configs and cloud stubs), `check_required_fields.py`, and `check_artifact_shape.py`. `docs-check.yml` only checks that both schema files are valid JSON.
- Generated outputs (`artifacts/packs/*`, `artifacts/registry.json`, `artifacts/saaq/`, `artifacts/manifests/`, `artifacts/handoff/`) are gitignored. Do not commit them.
- `magere-corinth-core` (`// SPDX-License-Identifier: Apache-2.0 OR MIT`) and `magere-grok-process` (`// SPDX-License-Identifier: GPL-3.0-or-later`) start every `.rs` file with an SPDX header; keep it on new files there. `magere-cli` and `magere-bridge` have none, so do not add one.

## Gotchas

- The first `cargo clippy` or `cargo test` compiles the full dependency tree (`jsonschema` among others) and took about 5 minutes here.
- `Cargo.lock` is tracked and the Dockerfile builds with `--locked`; commit lockfile changes with dependency changes.
- No crate sets `publish`, `exclude`, or `include`, so there is no packaging list to update. Never run `cargo publish`.
- `.claude/` is not in `.gitignore`; do not commit local agent state there.
- Edit only what the task needs.
