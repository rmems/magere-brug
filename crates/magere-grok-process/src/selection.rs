// SPDX-License-Identifier: GPL-3.0-or-later
//! Tensor selection — classify tensors into precision tiers.


/// Classification of a single tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorClass {
    Router,
    Norm,
    Expert,
    Projection,
    Embedding,
    Other,
}

/// Classify a tensor name into its structural role.
pub fn classify(name: &str) -> TensorClass {
    let lower = name.to_lowercase();
    if lower.contains("router") || lower.contains("gate") {
        TensorClass::Router
    } else if lower.contains("norm") || lower.contains("ln") {
        TensorClass::Norm
    } else if lower.contains("expert") || lower.contains("mlp") {
        TensorClass::Expert
    } else if lower.contains("proj") || lower.contains("projection") {
        TensorClass::Projection
    } else if lower.contains("embed") || lower.contains("token") {
        TensorClass::Embedding
    } else {
        TensorClass::Other
    }
}
