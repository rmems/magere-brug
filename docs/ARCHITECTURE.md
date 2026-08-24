# magere-brug Manifest Architecture

## Overview

magere-brug is a **Spiking Adaptive Activity Quantization (SAAQ) lab** that owns the artifact registry, manifest system, and experiment recipes for selected MoE/SNN quantization experiments. This document describes the manifest format, batch structure, crate layout, and reproducibility guarantees.

The repository name is intentionally humorous (a Dutch bridge engineering reference). Manifest handoff to downstream pipelines (like `combine-for-AI`) is managed via standardized JSON files, but magere-brug itself focuses on SAAQ orchestration: preparing artifacts, defining recipes, and validating SNN-based quantization pipelines.

This is **not** a general model-training framework. For model training, see `rmems/agoge-forger`.

## Stack Architecture

### Rust Responsibilities

The Rust workspace owns:

- **`magere-cli`** — Manifest validation, artifact registry, checksums, path normalization, run metadata, and handoff files.
- **`magere-corinth-core`** — CPU-only SNN pipeline components (`TelemetryEncoder`, `SparseGifHiddenLayer`, `Projector`, `SnnLatentCalibrator`) used for SAAQ validation.
- **`magere-grok-process`** — Grok-1 specific weight packing, ternary quantization, and manifest parsing utilities.
- **`magere-bridge`** — Placeholder/WIP crate for future cross-crate glue or external bridge logic. Currently a minimal binary stub.

### Python Responsibilities

Python helpers in `scripts/` own:

- **HuggingFace model loading** and introspection
- **Safetensors header inspection** for dtype, shard, metadata extraction
- **Model conversion scripts** (e.g., to GGUF format)
- **Calibration dataset helpers** and manifest snippet generation

### CUDA/Kernels

**NOT in magere-brug.** Neuromorphic inference kernels and model-specific inventory tools live in:

- `myelin-accelerator` — Blackwell-first CUDA kernels for neuromorphic inference
- `xai-dissect` — static analysis of Grok-family checkpoints

magere-brug **calls** these backends later when execution is needed; it does not own kernel code.

---

## Manifest Format

### Schema Location

`schemas/model_manifest.schema.json` — Complete JSON Schema v7 definition for validation.

### Core Structure

```json
{
  "metadata": {
    "schema_version": 1,
    "created_at": "ISO 8601 timestamp",
    "manifest_id": "model_slug-version",
    "description": "Human-readable description"
  },
  "model": {
    "slug": "stable_identifier",
    "name": "Human-readable name",
    "family": "olmoe|gemma|deepseek|...",
    "parameter_count": {
      "active": 7000000000,
      "total": 8000000000
    },
    "architecture": "dense|moe",
    "moe_layout": {
      "expert_count": 8,
      "capacity_factor": 1.25,
      "routing_strategy": "top-2"
    }
  },
  "source_artifact": {
    "format": "safetensors|gguf|hf_repo|local_dir",
    "path": "/local/path/or/hf_id",
    "source_url": "https://...",
    "checksum": {
      "sha256": "hex_string",
      "md5": "hex_string"
    },
    "dtype_summary": "fp32|fp16|bf16|int8|int4|mixed",
    "size_bytes": 14000000000,
    "shard_info": {
      "shard_count": 2,
      "shard_size_bytes": 7000000000,
      "shard_paths": ["model-00001.safetensors", "model-00002.safetensors"]
    }
  },
  "generated_artifact": {
    "format": "goz1|gguf|ternary|binary",
    "status": "planned|running|success|failed|skipped",
    "path": "/output/artifact.goz1",
    "version": 1,
    "checksum": { "sha256": "...", "md5": "..." },
    "size_bytes": 4000000000,
    "timestamp": "ISO 8601",
    "source_lineage": {
      "manifest_id": "parent-manifest-id",
      "path": "/models/source.gguf"
    },
    "tensor_summary": {
      "tensor_count": 128,
      "f16_count": 16,
      "ternary_count": 112
    }
  },
  "quantization": {
    "method": "ternary|binary|gguf|saaq|none",
    "bits": 1|2|3|4|8,
    "group_size": 128,
    "calibration_dataset": "wikitext|math_logic|...",
    "calibration_config_path": "/configs/quantization/..."
  },
  "backend_compatibility": {
    "safetensors": { "supported": true, "status": "proven|testing|planned|not_applicable" },
    "gguf": { "supported": true, "status": "..." },
    "goz1": { "supported": true, "status": "..." },
    "myelin_accelerator": { "supported": true, "status": "...", "kernel_types": ["binary", "ternary", "saaq"] }
  },
  "saaq_experiment": {
    "saaq_version": "saaq_v1_5|legacy|...",
    "telemetry_source": "csv_telemetry|csv_re4_path_tracing_telemetry|...",
    "routing_entropy_metrics": {
      "expert_utilization": 0.87,
      "entropy_score": 2.14,
      "load_balance_score": 0.92
    }
  },
  "benchmark_linkage": {
    "pipeline_id": "downstream_pipeline_id",
    "status": "pending|queued|running|completed|failed",
    "results_path": "/results/..."
  }
}
```

### Required Fields

- `metadata.schema_version` ≥ 1
- `metadata.created_at` — ISO 8601 timestamp
- `metadata.manifest_id` — Unique identifier
- `model.slug` — Stable model identifier (lowercase, underscores)
- `model.name` — Human-readable name
- `model.family` — Model family classification
- `model.parameter_count.active` — Active parameter count (≥ 0)
- `model.architecture` — dense or moe
- `source_artifact.format` — safetensors, gguf, hf_repo, or local_dir
- `source_artifact.path` — Filesystem path or HF repo ID

### Optional Fields

- `quantization` — Omit if source is unquantized
- `generated_artifact` — Populated after packing/quantization (prefer `format: "goz1"`)
- `backend_compatibility` — Backend support matrix
- `saaq_experiment` — SAAQ metadata (when applicable)
- `benchmark_linkage` — Downstream pipeline integration

### GOZ1 generated artifacts

**GOZ1** is the hybrid packed-weight format written by `magere-grok-process` (file magic `GOZ1`, ternary tensors + FP16 passthrough). It is first-class in manifests:

| Field | Role |
|-------|------|
| `format: "goz1"` | Identifies a GOZ1 pack |
| `version` | Pack format version (aligns with `GOZ1_VERSION`, currently `1`) |
| `path` / `checksum` / `size_bytes` | On-disk location and integrity |
| `source_lineage` | Parent manifest id and source path used for packing |
| `tensor_summary` | Optional `tensor_count`, `f16_count`, `ternary_count` |

Primary path: **packable source → ternary pack (`magere-grok-process`) → GOZ1 → SAAQ (`magere-corinth-core`) → handoff (`combine-for-AI`)**. AWQ and GPTQ are not supported.

`magere-grok-process` currently accepts **safetensors** and **npy_dir** as packer `InputFormat`s. NPY directories are recorded in manifests as `source_artifact.format: "local_dir"` (`npy_dir` is **not** a valid `source_artifact.format`) and mapped to `InputFormat::NpyDir` before packing. GGUF remains a first-class *registry* source format for local routing and SAAQ, but is not a direct packer input yet.

### Recipe registration (thin)

`schemas/recipe.schema.json` defines recipes that reference manifests and GOZ1 packs:

- `type`: `register` | `goz1_pack` | `ternary_pack` | `saaq`
- `inputs.source_manifest` — path or registry id of the source manifest
- `inputs.goz1_ref` — path or registry id of a registered GOZ1 pack
- `outputs.generated_format` — prefer `goz1`
- `outputs.output_dir` — directory for pack or SAAQ run outputs
- `saaq` — SAAQ runner configuration; only valid on `type: "saaq"` recipes (see [SAAQ Runner](#saaq-runner) below)

Examples: `configs/recipes/goz1-ref-example.json` (reference-only), `configs/recipes/saaq-example.json` (runnable). The packing CLI is tracked separately; the SAAQ runner is implemented — see below.

---

## Batch Structure

magere-brug uses a **batch-based organization** inspired by corinth-canal for model onboarding:

### Batch A: Local GGUF Routing Targets

**File:** `configs/models/batch-a-gguf-routing.json`

Models loadable locally via GGUF format for SAAQ latent calibration runs:

- `olmoe_baseline` — OLMoE MoE baseline
- `gemma4_26b_a4b_iq4_nl` — Gemma-4 MoE model
- `deepseek_coder_v2_lite_q6_k_l` — DeepSeek-Coder-V2-Lite MoE model
- `llama_3_2_dark_champion_q5_k_m` — Llama MoE variant
- `zaya1_8b_q8_0` — Zaya-1 dense model
- `glm_4_6v_flash_q8_0` — GLM-4.6V-Flash dense model
- `kimi_vl_a3b_q6_k` — Kimi-VL-A3B MoE model
- `marco_nano_base_q8_0` — Marco-Nano-Base dense model

### Batch B: Local Safetensors Manifest Inspection

**File:** `configs/models/batch-b-safetensors.json`

Local safetensors models for header inspection and manifest generation. Router path remains GGUF-backed:

- `redpajama_incite_7b_chat` — Baseline safetensors model
- `nemotron_3_nano_4b` — NVIDIA Nemotron dense model
- `granite_3_1_3b_a800m` — IBM Granite MoE model
- `trinity_nano_base` — Arcee Trinity MoE model

### Batch B-local: Local Directory Checkpoints

**File:** `configs/models/batch-b-local-dir.json`

Local directory (non-safetensors) checkpoint models for inspection and routing experiments. These entries use `source_format: "local_dir"` and include `shard_count` and `dtype` fields:

- `phi_tiny_moe_instruct` — Microsoft Phi-tiny MoE instruct checkpoint (BF16, 2 shards)
- `moonlight_16b_a3b_bnb4bit` — Moonlight-16B-A3B BNB 4-bit quantized checkpoint

### Batch C: Cloud Model Metadata Stubs

**Directory:** `configs/models/cloud/`

Cloud provider stubs with **no real credentials, endpoints, or keys**. All stubs marked:

```json
{
  "stub": true,
  "status": "stub",
  "enabled": false,
  "requires_secrets": false
}
```

> **Note:** Active (non-stub) cloud configs must set `"requires_secrets": true`.

**Files:**

- `nvcf-nim.json` — NVIDIA NIM (NeMo Inference Microservice)
- `vertex-ai.json` — Google Cloud Vertex AI
- `openai-compat.json` — OpenAI-compatible API providers (vLLM, LiteLLM, etc.)
- `local-vllm.json` — Local vLLM server development stubs
- `local-llamacpp.json` — Local llama.cpp server development stubs

**Security Note:** Stubs use placeholder tokens like `[SET_API_KEY]` and environment variable references. Never expose real credentials in PR code.

---

## Example Manifests

Located in `manifests/examples/`:

### 1. redpajama-incite-7b-chat.json

**Purpose:** Baseline safetensors model for ternary→GOZ1 packing and SAAQ handoff

- Format: Safetensors
- Architecture: Dense
- Parameters: 7B active
- Quantization Method: ternary (planned GOZ1 pack)
- Backend: Safetensors proven; GOZ1 and GGUF planned

### 2. olmoe-1b-7b-instruct.json

**Purpose:** MoE model with GGUF local routing target

- Format: GGUF
- Architecture: MoE (8 experts, 1.25 capacity factor)
- Parameters: 7B active, 8B total
- Quantization: F16 (no quantization)
- SAAQ Experiment: SAAQ v1.5, CSV RE4 path tracing telemetry
- Backend Compatibility: GGUF proven; GOZ1 and safetensors planned
- Status: Completed benchmark run (olmoe_baseline_csv_re4_control)

### 3. deepseek-coder-v2-lite.json

**Purpose:** DeepSeek-Coder-V2-Lite quantization model

- Format: GGUF (Q6_K variant)
- Architecture: MoE (64 experts, top-6 routing)
- Parameters: 16B total / 2.4B active
- Quantization: GGUF Q6_K (existing)
- Backend: GGUF proven; GOZ1 planned
- Use Case: Code model quantization track

### 4. grok-1-future-plan.json

**Purpose:** Future planning entry for Grok-1

- **Status:** Source-only, no execution
- **Management:** Via grok-ozempic + xai-dissect (separate repos)
- **Target pack:** GOZ1 (planned), ternary method
- **Backend:** myelin-accelerator (ternary, binary, SAAQ kernels planned)
- **Format:** Local directory (checkpoint)
- **Architecture:** MoE (256 experts, expert choice routing)

### 5. goz1-pack-example.json

**Purpose:** Full example of a registered GOZ1-generated artifact

- Source: Safetensors (pack-compatible input for `magere-grok-process`)
- Generated: `format: "goz1"`, version 1, path, checksum, `source_lineage`, `tensor_summary`
- Quantization method: ternary
- Backend: `goz1` + `myelin_accelerator` planned

---

## SAAQ Runner

`magere run-saaq <RECIPE>` executes a `type: "saaq"` recipe against the CPU-only pipeline in `magere-corinth-core`. It is a **validation** pass over a telemetry stream — it observes how the SNN pipeline responds and records the SAAQ latent trajectory. It does not touch model weights, and there is no training loop anywhere in it.

### Flow

```
configs/recipes/saaq-example.json          (recipe: refs + SAAQ config)
  ↓ validate: type, projection mode, update rule, thresholds, snn_steps, input refs on disk
telemetry stream                            (synthetic ramp, or replayed from a telemetry CSV)
  ↓ TelemetryFunnel::encode_snapshot        (TelemetryEncoder → ternary events → signed split banks → sparse GIF hidden layer)
FunnelActivity                              (ternary_events, spike_train, potentials, iz_potentials)
  ↓ Projector::project                      (spike train + potentials → dense embedding [2048])
embedding
  ↓ deterministic expert weights            (SURROGATE, not the model's router — softmax over per-slice means of `num_experts` embedding slices)
ModelOutput                                 (spike train, firing rates, membranes, embedding, expert weights, selected experts)
  ↓ SnnLatentCalibrator / SnnDualLatentCalibrator
SnnLatentSnapshot per tick
  ↓ SnnLatentCsvExporter
<output_dir>/latent_telemetry.csv + <output_dir>/run_manifest.json
```

### Recipe `saaq` block

| Field | Default | Role |
|-------|---------|------|
| `projection_mode` | `SpikingTernary` | `ProjectionMode`: `RateSum`, `TemporalHistogram`, `MembraneSnapshot`, `SpikingTernary` |
| `snn_steps` | `20` | SNN time-steps expanded per telemetry snapshot |
| `thresholds` | `[1.0, 5.0, 1.0, 5.0]` | Exactly 4 `TelemetryEncoder` thresholds: `gpu_temp_c`, `gpu_power_w`, `cpu_tctl_c`, `cpu_package_power_w` |
| `update_rule` | `LegacyV1_0` | `SaaqUpdateRule`: `LegacyV1_0` or `SaaqV1_5SqrtRate` |
| `dual_rule` | `false` | Observe both rules through `SnnDualLatentCalibrator`, filling the `*_legacy_*` and `*_v15_*` columns |
| `num_experts` | `8` | **Surrogate router, not the model's own.** Slices the embedding is split across to derive the routing distribution behind `routing_entropy`. Capped at the embedding dim (2048) so every expert owns at least one position |
| `top_k` | `1` | **Recorded-only provenance.** Experts listed in `selected_experts` each tick. The calibrator reads only `expert_weights`, so `top_k` cannot change the CSV — runs differing only in `top_k` are byte-identical |
| `telemetry.source` | `synthetic` | `synthetic` or `csv` |

> **The expert weights are a placeholder, and `routing_entropy` is near-constant.**
> `expert_weights` is *not* the source model's MoE router. It is a surrogate derived
> entirely from telemetry-driven SNN activity — no model weights are read — by softmaxing
> the per-slice means of `num_experts` contiguous equal-width slices of the projector
> embedding. A run manifest that also names a real MoE `source_manifest` and a GOZ1 ref
> would otherwise read as if the column came from the model itself.
>
> Its dynamic range is correspondingly narrow. Over the checked-in
> `configs/recipes/saaq-example.json` run, `SpikingTernary` leaves the embedding exactly
> zero on 10 of 16 ticks (giving exactly uniform weights, entropy `0.99999988`), and on the
> 6 non-zero ticks only 16–175 of 2048 positions fire. Entropy is quadratically flat near
> its maximum, so the whole run spans `0.991067 → 1.000000` — under 1% of `[0, 1]`. The
> `0.20 * routing_entropy − 0.18` term in `LegacyV1_0` is therefore a near-constant
> `+0.02`. The math is correct, but treat `routing_entropy` as a recorded diagnostic, not
> a live routing signal: a downstream symbolic-regression fit would see a constant column.

`telemetry.source: "synthetic"` takes `ticks`, `tick_interval_ms`, `start_timestamp_ms`, and `start` / `delta` channel blocks. `telemetry.source: "csv"` takes `path` instead, and rejects the synthetic-only knobs rather than silently ignoring them. The CSV must carry `timestamp_ms`, `gpu_temp_c`, `gpu_power_w`, `cpu_tctl_c` and `cpu_package_power_w` columns, in any order.

Input refs (`inputs.source_manifest`, `inputs.goz1_ref`) must resolve to a file on disk. A reference is tried as written (absolute, or relative to the current directory) and then against the recipe's own directory and each of its parents, so a repo-root-relative reference in a nested recipe works from any working directory.

### Outputs

**`latent_telemetry.csv`** — one row per tick, in the exact `SnnLatentCsvExporter` column layout:

```
timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,saaq_delta_q_prev,saaq_delta_q_target,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w,saaq_delta_q_legacy_prev,saaq_delta_q_legacy_target,saaq_delta_q_v15_prev,saaq_delta_q_v15_target
```

**`run_manifest.json`** — the reproducibility record: recipe id/path, resolved input refs (each canonicalised and pinned by SHA256), every effective SAAQ parameter (including the expert-weight scheme), telemetry parameters — with the input CSV's SHA256 for a `csv` source — tick count, output CSV path and SHA256, crate versions, and a `created_at` timestamp.

### Determinism

Replaying a recipe reproduces `latent_telemetry.csv` byte for byte on any machine with the
same floating-point behaviour (the pipeline goes through libm `exp`/`ln`, which are not
correctly-rounded and may differ by an ULP across platforms/libm versions). Debug and
`--release` builds agree on the same machine. Within that scope:

- the `synthetic` source is a pure function of the tick index — channel `c` at tick `i` is `start.c + delta.c * i`, with no RNG and no wall clock;
- `timestamp_ms` comes from `start_timestamp_ms + tick_interval_ms * i`, never from the system clock;
- expert weights are a pure function of the projector embedding;
- the projector and funnel weight matrices are fixed, deterministically initialised constants of `magere-corinth-core`;
- the only wall-clock value the runner emits (`created_at`) lives in `run_manifest.json`, deliberately kept out of the CSV so two runs can be diffed directly.

Because the ramp is sub-threshold per tick and the encoder only re-baselines on a threshold crossing, a small `delta` yields a *periodic* spike pattern rather than a constant one — which is what makes a short ramp a useful validation signal.

---

## Reproducibility Guarantees

A manifest guarantees model reproducibility if:

1. **Source artifact is specified completely:**
   - Format, path, URL
   - SHA256/MD5 checksums
   - For safetensors: shard paths, shard sizes, dtype summary

2. **Quantization parameters are captured:**
   - Method (ternary, binary, gguf, saaq, none)
   - Bit width, group size when applicable
   - Calibration dataset reference
   - Calibration config path

3. **Backend compatibility is documented:**
   - Which formats are supported (safetensors, gguf, goz1, myelin_accelerator)
   - Status of each backend (proven, testing, planned, not_applicable)

4. **Run metadata is recorded:**
   - SAAQ version and telemetry source
   - Routing entropy metrics (if applicable)
   - Benchmark linkage and results path

### Example Reproducibility Chain

```
Source Artifact for packing (safetensors | local_dir → InputFormat::NpyDir)
  ↓ [magere verify <artifact> <sha256>]
Ternary pack via magere-grok-process (skeleton: header/table layout; tensor load TBD)
  ↓ [GOZ1 magic, version 1 required on successful packs, tensor table]
Generated Artifact (GOZ1)
  ↓ [register in manifest: path, checksum, lineage]
SAAQ validation (magere-corinth-core) — may also start from GGUF registry sources
  ↓ [magere run-saaq <recipe> → latent_telemetry.csv + run_manifest.json]
Benchmark / reporting (combine-for-AI)
  ↓
Results & Analysis
```

---

## Rust CLI Usage

The `magere` CLI (from `crates/magere-cli`) provides manifest management:

### Validate a Manifest

```bash
cargo run --bin magere -- validate manifests/examples/olmoe-1b-7b-instruct.json
```

**Output:** ✓ Manifest validation passed or ✗ Error with details

### Inspect a Manifest

```bash
cargo run --bin magere -- inspect manifests/examples/olmoe-1b-7b-instruct.json
```

**Output:** Human-readable manifest fields (name, family, slug, architecture, parameters, source, quantization, backend support)

### Register a Model

```bash
cargo run --bin magere -- register manifests/examples/redpajama-incite-7b-chat.json --registry configs/models/registry.json
```

**Output:** Registered model slug added to registry

### Verify Artifact Checksum

```bash
cargo run --bin magere -- verify /models/olmoe/OLMoE-1B-7B-0125-Instruct-F16.gguf abc123def456...
```

**Output:** ✓ Checksum verified or ✗ Checksum mismatch

### Run a SAAQ Recipe

```bash
cargo run --bin magere -- run-saaq configs/recipes/saaq-example.json
cargo run --bin magere -- run-saaq configs/recipes/saaq-example.json --output-dir /saaq/olmoe-run-01
```

`--output-dir` overrides the recipe's `outputs.output_dir`; one of the two must be present. See [SAAQ Runner](#saaq-runner) for the recipe fields and the determinism guarantees.

**Output:** ✓ Run summary plus the paths of `latent_telemetry.csv` (with its SHA256) and `run_manifest.json`

---

## Python Helper Scripts

Located in `scripts/`:

### inspect_safetensors.py

Extract metadata from safetensors files:

```bash
python scripts/inspect_safetensors.py /models/redpajama/INCITE-7B-Chat/model-00001.safetensors
```

**Returns:** dtype_summary, tensor_count, shard info, file size

### register_gguf.py

Register GGUF models and generate manifest snippets:

```bash
python scripts/register_gguf.py /models/olmoe/OLMoE-1B-7B-0125-Instruct-F16.gguf
```

**Returns:** GGUF version, tensor count, inferred quantization format

### GOZ1 packing (Rust)

**`magere-grok-process`** owns the GOZ1 layout (magic, version, tensor table, stream/pack builders). The end-to-end packer (`run_quantize`) is still a **skeleton**: it builds a structurally valid pack shell but does not yet load real tensor weights (placeholder shapes/data). Do not treat skeleton output as a production checkpoint. Recipe-driven pack CLI is tracked separately; manifests register completed GOZ1 packs as first-class `generated_artifact` entries when a real pack path exists.

### Unit Tests

All Python helpers include pytest-based unit tests:

```bash
cd scripts && python -m pytest tests/test_manifest.py -v
```

**Coverage:**
- Manifest loading and validation
- Required field presence
- Checksum field shape
- GGUF/Safetensors source format acceptance
- GOZ1 generated-artifact structure

---

## Integration with Downstream Pipelines

### Manifest Handoff Format

magere-brug creates standardized JSON handoff files for downstream consumers (e.g., `combine-for-AI`):

```json
{
  "manifest_ref": "olmoe-1b-7b-instruct-v1",
  "model_slug": "olmoe_baseline",
  "source_artifact": {
    "path": "/models/olmoe/OLMoE-1B-7B-0125-Instruct-F16.gguf",
    "checksum": "sha256:..."
  },
  "benchmark_linkage": {
    "pipeline_id": "downstream_pipeline_id",
    "status": "ready"
  }
}
```

### SAAQ Integration

Manifests track SAAQ experiment metadata:

- **SAAQ Version:** saaq_v1_5, legacy, etc.
- **Telemetry Source:** csv_telemetry, csv_re4_path_tracing_telemetry, etc.
- **Routing Entropy Metrics:** expert utilization, entropy score, load balance score
- **Results Path:** Location of benchmark output

---

## Limitations & Future Work

### Done (foundation)

- ✓ JSON Schema definition (including first-class GOZ1)
- ✓ Rust CLI skeleton (parsing, validation, registry)
- ✓ Python helper script stubs (GGUF/safetensors inspection)
- ✓ Example manifests (including GOZ1 pack example)
- ✓ Thin recipe schema for GOZ1 refs
- ✓ Recipe-driven SAAQ runner (`magere run-saaq`) with deterministic telemetry + run manifest
- ✓ Batch A/B structure + Cloud stubs
- ✓ Documentation (primary path ternary → GOZ1 → SAAQ)

### Explicitly out of scope for this lab

- AWQ / GPTQ quantization paths (removed; not on the primary path)
- CUDA kernels for neuromorphic inference (see `myelin-accelerator`)
- Training loops (see `rmems/agoge-forger`)

### Next pipeline work

- Recipe-driven ternary pack → GOZ1 via `magere-grok-process`
- SAAQ runs driven from real hardware telemetry captures rather than synthetic ramps
- myelin-accelerator kernel invocation from handoff manifests
- Cloud backend integration (NIM, Vertex AI, etc.) when needed
- Extended batch model onboarding

---

## References

- **Batch Structure:** Inspired by corinth-canal's batch model (Batch A/B/C)
- **SAAQ:** Spiking Adaptive Activity Quantization framework
- **Related Repos:**
  - `rmems/agoge-forger` — model-training forge
  - `corinth-canal` — reference SAAQ implementation
  - `myelin-accelerator` — Blackwell-first CUDA kernels for neuromorphic inference
  - `xai-dissect` — static analysis of Grok-family checkpoints
  - `combine-for-AI` — neutral benchmark harness for quantization experiments
