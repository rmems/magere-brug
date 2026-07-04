// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Error types for `magere-corinth-core`.

use thiserror::Error;

/// Unified error type for the SNN block.
#[derive(Debug, Error)]
pub enum SnnError {
    /// A requested configuration is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Input slice had the wrong length.
    #[error("input length mismatch: expected {expected}, got {got}")]
    InputLengthMismatch { expected: usize, got: usize },

    /// The SNN state is inconsistent, e.g. a method was called before forward.
    #[error("state error: {0}")]
    StateError(String),

    /// An I/O error occurred, usually while writing diagnostics.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, SnnError>;
