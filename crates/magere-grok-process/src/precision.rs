// SPDX-License-Identifier: GPL-3.0-or-later
//! Precision decision engine — assign tiers per tensor.

use crate::types::TensorPrecision;

/// Decide precision for a tensor based on manifest classification.
pub fn decide_precision(
    tensor_name: &str,
    manifest: &crate::manifest::DissectManifest,
) -> TensorPrecision {
    // Preserve list wins
    if manifest
        .preserve
        .iter()
        .any(|p| glob_match(&p.name, tensor_name))
    {
        return TensorPrecision::Preserve;
    }
    // FP16 list
    if manifest
        .fp16
        .iter()
        .any(|f| glob_match(&f.name, tensor_name))
    {
        return TensorPrecision::Fp16;
    }
    // Default to ternary
    TensorPrecision::TernarySnN
}

/// Simple glob-style matching.
fn glob_match(pattern: &str, text: &str) -> bool {
    text.contains(pattern)
}
