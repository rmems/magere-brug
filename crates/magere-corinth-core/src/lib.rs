// SPDX-License-Identifier: Apache-2.0 OR MIT
//! # magere-corinth-core
//!
//! Proven CPU-only SNN pipeline components extracted from `corinth-canal`.
//!
//! ## Runtime pipeline
//!
//! ```text
//! TelemetrySnapshot
//!        │
//!        ▼  TelemetryEncoder
//! ternary telemetry events (+1 / 0 / -1 per channel)
//!        │
//!        ▼  SignedSplitBankBridge
//! input spike train
//!        │
//!        ▼  SparseGifHiddenLayer
//! hidden spike train + membrane potentials
//!        │
//!        ▼  Projector
//! dense embedding [EMBEDDING_DIM = 2048]
//!        │
//!        ▼  SnnLatentCalibrator
//! SAAQ latent calibration / telemetry export
//! ```
//!
//! This crate intentionally contains **no CUDA code** — GPU kernels live in
//! `corinth-canal` and `myelin-accelerator`. All modules here are pure CPU
//! and suitable for use in any environment without a GPU toolkit.

pub mod error;
pub mod funnel;
pub mod latent;
pub mod metric;
pub mod projector;
pub mod telemetry;
pub mod types;

pub use error::{HybridError, Result};
pub use funnel::{
    FUNNEL_HIDDEN_NEURONS, FUNNEL_INPUT_NEURONS, FunnelActivity, SignedSplitBankBridge,
    SparseGifHiddenLayer, TelemetryFunnel,
};
pub use latent::{
    SaaqUpdateRule, SnnDualLatentCalibrator, SnnLatentCalibrator, SnnLatentCsvExporter,
    SnnLatentSnapshot,
};
pub use telemetry::TelemetryEncoder;
pub use types::{
    CloudModelSpec, EMBEDDING_DIM, ModelArchitectureClass, ModelFamily, ModelTarget,
    TelemetrySnapshot, ProjectionMode, RoutingMode, ModelConfig, ModelOutput,
    CheckpointFormat,
};
