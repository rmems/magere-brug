# magere-brug

**Spiking Adaptive Activity Quantization (SAAQ) lab** for selected MoE/SNN experiments.

`magere-brug` owns the artifact registry, recipe system, and experiment manifests needed to prepare and track SAAQ quantization experiments. It is **not** a general model-training framework. For model training, see [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger).

## Scope

Owns:
- Model artifact registry and manifests
- SAAQ quantization recipes
- Calibration dataset configuration
- Per-experiment run logs and handoff files
- CPU-only SNN pipeline components (`magere-corinth-core`) for SAAQ validation

Does not own:
- CUDA kernels for neuromorphic inference (see `myelin-accelerator`)
- Benchmark scoring harnesses (see `combine-for-AI`)
- General model-training code (see `rmems/agoge-forger`)
- Grok-family checkpoint analysis (see `xai-dissect`)

## Related repositories

- [`corinth-canal`](https://github.com/rmems/corinth-canal) — reference SAAQ implementation
- [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger) — model-training forge
- [`xai-dissect`](https://github.com/rmems/xai-dissect) — static analysis of Grok-family checkpoints
- [`myelin-accelerator`](https://github.com/rmems/myelin-accelerator) — Blackwell-first CUDA kernels for neuromorphic inference
- [`combine-for-AI`](https://github.com/rmems/combine-for-AI) — neutral benchmark harness for quantization experiments

## Quick start

```bash
cargo build --workspace
cargo test --workspace
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the manifest format and crate layout.
