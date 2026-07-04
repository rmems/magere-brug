# Agent Guidance for magere-brug

## Project purpose

`magere-brug` is a **Spiking Adaptive Activity Quantization (SAAQ) lab**. It owns the artifact registry, manifest system, recipes, and CPU-only SNN validation components used to prepare and track SAAQ quantization experiments.

This is **not** a general model-training repository. For model training, see `rmems/agoge-forger`.

## Scope boundaries

In scope:
- Model manifests and artifact registry (`crates/magere-cli`)
- SAAQ recipes and experiment tracking
- CPU-only SNN pipeline components (`crates/magere-corinth-core`)
- Calibration config and per-run handoff files
- Docker packaging and CI for the above

Out of scope:
- CUDA kernels for neuromorphic inference (live in `myelin-accelerator`)
- Benchmark scoring harnesses (live in `combine-for-AI`)
- General model-training code (live in `rmems/agoge-forger`)
- Grok-family checkpoint analysis (live in `xai-dissect`)

## Key constraints

- Do not reframe `magere-corinth-core` as a generic trainable SNN block. It is an SAAQ validation pipeline: `TelemetrySnapshot` → `TelemetryEncoder` → spikes → hidden GIF layer → `Projector` → SAAQ calibration.
- Do not introduce training-loop concepts (loss functions, optimizers, gradient descent, dataloaders) unless the task explicitly asks for them and the work is clearly scoped as a bridge to `agoge-forger`.
- Preserve the telemetry-oriented abstraction in `magere-corinth-core` unless explicitly asked to change it.
- All changes go through PRs; do not push directly to `main`.

## Related repositories

- `corinth-canal` — reference SAAQ implementation
- `rmems/agoge-forger` — model-training forge
- `myelin-accelerator` — Blackwell-first CUDA kernels for neuromorphic inference
- `combine-for-AI` — neutral benchmark harness for quantization experiments
- `xai-dissect` — static analysis of Grok-family checkpoints

## Build and validation

```bash
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Docs-only changes do not require full `cargo` validation, but running the commands above is still encouraged when the change touches crate docs or module-level doc comments.
