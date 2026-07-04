// SPDX-License-Identifier: Apache-2.0 OR MIT
//! # magere-corinth-core
//!
//! Trainable CPU-only SNN building block for model-training pipelines.
//!
//! This crate provides a composable spiking neural network layer that can be
//! used as a building block when training models with SNN, ANN, or hybrid
//! architectures. It intentionally contains **no CUDA code** — GPU kernels
//! live in `corinth-canal` and `myelin-accelerator`. All modules here are pure
//! CPU and suitable for use in any environment without a GPU toolkit.
//!
//! ## Core pipeline
//!
//! ```text
//! SpikeSample (spike_train)
//!        │
//!        ▼  GifHiddenLayer
//! hidden spike train + membrane potentials
//!        │
//!        ▼  SpikeToDenseProjector
//! dense embedding [embedding_dim]
//! ```
//!
//! The [`SnnBlock`] type composes the hidden layer and projector into a single
//! trainable unit. The projector weights and bias are learnable; the hidden
//! layer is a fixed sparse reservoir in this version.

pub mod block;
pub mod error;
pub mod funnel;
pub mod latent;
pub(crate) mod metric;
pub mod projector;
pub mod types;

pub use block::SnnBlock;
pub use error::{Result, SnnError};
pub use funnel::{GifHiddenLayer, HiddenActivity};
pub use latent::{SnnMetrics, compute_metrics};
pub use projector::SpikeToDenseProjector;
pub use types::{
    DEFAULT_DIM, ProjectionGradients, ProjectionMode, SnnBlockConfig, SnnBlockOutput, SpikeSample,
};
