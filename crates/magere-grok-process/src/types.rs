// SPDX-License-Identifier: GPL-3.0-or-later
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Controls which precision is applied to a given tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorPrecision {
    /// Two-bit ternary {-1, 0, +1} with saliency-gated GIF threshold.
    TernarySnN,
    /// Keep original FP16 — used for MoE routing gates.
    Fp16,
    /// Routing-critical / no-touch tier. Identical on-disk encoding to
    /// Fp16, but signals "must never be ternarized".
    Preserve,
}

/// Weight container layout for input sources.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputFormat {
    /// Hugging Face–style `*.safetensors` shards.
    #[default]
    Safetensors,
    /// Directory of per-tensor `*.npy` files (JAX/Flax export).
    NpyDir,
}

/// Configuration for the out-of-core quantization pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantizeConfig {
    pub input_dir: String,
    pub output_path: String,
    /// GIF saliency threshold ratio.
    pub gif_threshold: f32,
    pub input_format: InputFormat,
    pub manifest_path: Option<PathBuf>,
    pub use_embedded_baseline: bool,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            input_dir: String::new(),
            output_path: String::new(),
            gif_threshold: 0.05,
            input_format: InputFormat::Safetensors,
            manifest_path: None,
            use_embedded_baseline: false,
        }
    }
}

/// Grok-1 hidden size.
pub const GROK1_HIDDEN_DIM: usize = 6144;
pub const GROK1_VOCAB_SIZE: usize = 131_072;
pub const GROK1_TENSOR_TOTAL: usize = 770;

/// GOZ1 format constants.
pub const GOZ1_MAGIC: &[u8] = b"GOZ1";
pub const GOZ1_VERSION: u32 = 1;
