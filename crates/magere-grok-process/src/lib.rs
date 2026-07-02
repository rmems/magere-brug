// SPDX-License-Identifier: GPL-3.0-or-later
//! # magere-grok-process
//!
//! Grok-1 quantization engine reimplemented from `grok-ozempic` logic.
//!
//! This crate provides the three-pass out-of-core quantization pipeline:
//! manifest → header → data. It converts heavyweight Grok checkpoints
//! into a spiking-friendly GOZ1 packed format using ternary SNN encoding
//! and FP16 passthrough where necessary.

pub mod artifact;
pub mod error;
pub mod manifest;
pub mod precision;
pub mod quantizer;
pub mod selection;
pub mod stream;
pub mod types;
pub mod weight_pack;

pub use error::{GrokProcessError, Result};
pub use types::{TensorPrecision, QuantizeConfig, InputFormat};
