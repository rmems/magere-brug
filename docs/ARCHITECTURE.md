# magere-brug Manifest Architecture

## Overview

magere-brug is a **model quantization lab** that owns the artifact registry, manifest system, and quantization recipes for selected MoE and quantization experiments. This document describes the manifest format, batch structure, and reproducibility guarantees.

The repository name is intentionally humorous (a Dutch bridge engineering reference) and is **not related to sports benchmarking**. Manifest handoff to downstream pipelines (like NFL-combine-for-AI) is managed via standardized JSON files, but magere-brug itself focuses purely on model quantization orchestration.

## Stack Architecture

### Rust Responsibilities

The Rust CLI (`magere-cli` crate) owns:

- **Manifest validation** against JSON Schema
- **Artifact registry** for tracking all registered models
- **Checksum/hash handling** for reproducibility (SHA256, MD5)
- **Path normalization** for consistent cross-platform paths
- **Run metadata** tracking and serialization
- **Handoff files** for downstream consumers

### Python Responsibilities

Python helpers in `scripts/` own:

- **HuggingFace model loading** and introspection
- **Safetensors header inspection** for dtype, shard, metadata extraction
- **Model conversion scripts** (e.g., to GGUF format)
- **Calibration dataset helpers** and manifest snippet generation

### CUDA/Kernels

**NOT in magere-brug.** Custom CUDA kernels live in:

- `myelin-accelerator` — Binary/ternary/SAAQ quantization kernels
- `xai-dissect` — Grok-1 specific inventory management

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
    "format": "awq|gguf|gptq|ternary|binary",
    "status": "planned|running|success|failed|skipped",
    "path": "/output/artifact",
    "checksum": { "sha256": "...", "md5": "..." },
    "size_bytes": 4000000000,
    "timestamp": "ISO 8601"
  },
  "quantization": {
    "method": "awq|gptq|ternary|binary|none",
    "bits": 1|2|3|4|8,
    "group_size": 128,
    "calibration_dataset": "wikitext|math_logic|...",
    "calibration_config_path": "/configs/quantization/..."
  },
  "backend_compatibility": {
    "safetensors": { "supported": true, "status": "proven|testing|planned|not_applicable" },
    "gguf": { "supported": true, "status": "..." },
    "awq": { "supported": true, "status": "..." },
    "gptq": { "supported": true, "status": "..." },
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
- `generated_artifact` — Populated after quantization run
- `backend_compatibility` — Backend support matrix
- `saaq_experiment` — SAAQ metadata (when applicable)
- `benchmark_linkage` — Downstream pipeline integration

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

**Purpose:** Baseline safetensors model for AWQ smoke testing

- Format: Safetensors
- Architecture: Dense
- Parameters: 7B active
- Quantization Method: AWQ (planned, 4-bit)
- Backend: Safetensors proven, AWQ/GGUF planned

### 2. olmoe-1b-7b-instruct.json

**Purpose:** MoE model with GGUF local routing target

- Format: GGUF
- Architecture: MoE (8 experts, 1.25 capacity factor)
- Parameters: 7B active, 8B total
- Quantization: F16 (no quantization)
- SAAQ Experiment: SAAQ v1.5, CSV RE4 path tracing telemetry
- Backend Compatibility: GGUF proven, safetensors/AWQ planned
- Status: Completed benchmark run (olmoe_baseline_csv_re4_control)

### 3. deepseek-coder-v2-lite.json

**Purpose:** DeepSeek-Coder-V2-Lite quantization model

- Format: GGUF (Q6_K variant)
- Architecture: MoE (64 experts, top-6 routing)
- Parameters: 16B total / 2.4B active
- Quantization: GGUF Q6_K (existing)
- Backend: GGUF proven, safetensors/AWQ/GPTQ planned
- Use Case: Code model quantization track

### 4. grok-1-future-plan.json

**Purpose:** Future planning entry for Grok-1

- **Status:** Source-only, no execution
- **Management:** Via grok-ozempic + xai-dissect (separate repos)
- **Backend:** myelin-accelerator (ternary, binary, SAAQ kernels planned)
- **Format:** Local directory (checkpoint)
- **Architecture:** MoE (256 experts, expert choice routing)
- **Not Active:** All cloud/local backends marked "not_applicable" except myelin_accelerator

---

## Reproducibility Guarantees

A manifest guarantees model reproducibility if:

1. **Source artifact is specified completely:**
   - Format, path, URL
   - SHA256/MD5 checksums
   - For safetensors: shard paths, shard sizes, dtype summary

2. **Quantization parameters are captured:**
   - Method (awq, gptq, ternary, binary, none)
   - Bit width, group size
   - Calibration dataset reference
   - Calibration config path

3. **Backend compatibility is documented:**
   - Which formats are supported
   - Status of each backend (proven, testing, planned, not_applicable)

4. **Run metadata is recorded:**
   - SAAQ version and telemetry source
   - Routing entropy metrics (if applicable)
   - Benchmark linkage and results path

### Example Reproducibility Chain

```
Source Artifact (safetensors)
  ↓ [checksum verification]
Calibration Dataset (wikitext-2)
  ↓ [config: 128 samples, 128 group size]
AWQ Quantization (4-bit)
  ↓ [myelin-accelerator kernel]
Generated Artifact (AWQ format)
  ↓ [handoff checksum]
Benchmark Pipeline (NFL-combine-for-AI)
  ↓ [SAAQ telemetry recording]
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

### quant_awq.py

Orchestrate AWQ quantization runs:

```bash
python scripts/quant_awq.py /models/redpajama /quantized/redpajama-awq-4bit
```

**Returns:** AWQ run plan, status tracking, manifest stub generation

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
- Quantization method support

---

## Integration with Downstream Pipelines

### Manifest Handoff Format

magere-brug creates standardized JSON handoff files for downstream consumers (e.g., NFL-combine-for-AI):

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

### Sprint (Before 2026-05-28)

**In Scope:**
- ✓ JSON Schema definition
- ✓ Rust CLI skeleton (parsing, validation, registry)
- ✓ Python helper script stubs
- ✓ 4 example manifests
- ✓ Batch A/B structure + Cloud stubs
- ✓ Documentation

**Out of Scope:**
- ✗ Full AWQ generation execution (schema only)
- ✗ myelin-accelerator kernel execution (schema fields only)
- ✗ Cloud backend real integration
- ✗ Full Grok-1 quantization execution
- ✗ GPTQ execution (placeholder fields only)

### Phase 2 (After Sprint)

- Full AWQ smoke test execution
- myelin-accelerator kernel invocation
- GPTQ comparison baseline runs
- Cloud backend integration (NIM, Vertex AI, etc.)
- Full Grok-1 ternary quantization
- Extended batch model onboarding

---

## References

- **Batch Structure:** Inspired by corinth-canal's batch model (Batch A/B/C)
- **SAAQ:** Spiking Adaptive Activity Quantization framework
- **Related Repos:**
  - xai-dissect — Grok-1 inventory management
  - corinth-canal — SAAQ routing and telemetry lab
  - myelin-accelerator — Custom CUDA quantization kernels
  - NFL-combine-for-AI — Downstream benchmark pipeline
