// SPDX-License-Identifier: GPL-3.0-or-later
//! Three-pass out-of-core quantization pipeline.
//!
//! Pass 1: Manifest — scan shards and record metadata only.
//! Pass 2: Header — write the GOZ1 header and tensor table.
//! Pass 3: Data — map each tensor, quantize, and stream into output.

use crate::error::Result;
use crate::manifest::DissectManifest;
use crate::precision::decide_precision;
use crate::types::{QuantizeConfig, TensorPrecision};
use crate::weight_pack::{PackBuilder, TENSOR_F16, TENSOR_TERNARY};

/// Run the three-pass quantization pipeline.
pub fn run_quantize(_config: &QuantizeConfig, manifest: &DissectManifest) -> Result<Vec<u8>> {
    // Pass 1: collect metadata (simplified — in real impl would scan files)
    let mut builder = PackBuilder::new();

    // For each tensor in manifest, decide precision and queue for packing
    for candidate in &manifest.ternary_candidates {
        let precision = decide_precision(&candidate.name, manifest);
        let (dtype, data) = match precision {
            TensorPrecision::Preserve | TensorPrecision::Fp16 => {
                // In real impl, would load actual tensor data
                (TENSOR_F16, vec![0u8; 4])
            }
            TensorPrecision::TernarySnN => {
                // In real impl, would load and quantize actual tensor data
                (TENSOR_TERNARY, vec![0u8; 4])
            }
        };
        builder.add_tensor(
            candidate.name.clone(),
            dtype,
            vec![1, 1], // placeholder shape
            data,
        );
    }

    builder.finalize()
}
