// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Public data types for `magere-corinth-core`.

use serde::{Deserialize, Serialize};

/// Default dimensionality for the SNN block input, hidden, and embedding layers.
pub const DEFAULT_DIM: usize = 4096;

/// Strategy used to convert spike activity into a dense embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProjectionMode {
    RateSum,
    TemporalHistogram,
    MembraneSnapshot,
    #[default]
    SpikingTernary,
}

/// A spike-encoded input sample.
///
/// `spike_train` holds the indices of active neurons per simulation step.
/// `potentials` is an optional per-neuron membrane state that can be provided
/// by the caller; when empty the layer starts from rest.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpikeSample {
    pub spike_train: Vec<Vec<usize>>,
    pub potentials: Vec<f32>,
}

impl SpikeSample {
    /// Create a spike sample from a per-step spike train.
    pub fn new(spike_train: Vec<Vec<usize>>) -> Self {
        Self {
            spike_train,
            potentials: Vec::new(),
        }
    }

    /// Number of simulation steps in the sample.
    pub fn steps(&self) -> usize {
        self.spike_train.len()
    }

    /// Number of distinct neurons that fired at least once.
    pub fn active_neuron_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for step in &self.spike_train {
            for &idx in step {
                seen.insert(idx);
            }
        }
        seen.len()
    }
}

/// Output of one forward pass through an SNN block.
#[derive(Debug, Clone, PartialEq)]
pub struct SnnBlockOutput {
    pub spike_train: Vec<Vec<usize>>,
    pub firing_rates: Vec<f32>,
    pub membrane_potentials: Vec<f32>,
    pub embedding: Vec<f32>,
}

/// Configuration for a trainable SNN block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnnBlockConfig {
    pub input_neurons: usize,
    pub hidden_neurons: usize,
    pub embedding_dim: usize,
    pub projection_mode: ProjectionMode,
    pub snn_steps: usize,
    pub fan_in: usize,
}

impl SnnBlockConfig {
    /// Create a square 4096-dimensional block, the default for this crate.
    pub fn dim_4096() -> Self {
        Self {
            input_neurons: DEFAULT_DIM,
            hidden_neurons: DEFAULT_DIM,
            embedding_dim: DEFAULT_DIM,
            projection_mode: ProjectionMode::default(),
            snn_steps: 20,
            fan_in: 4,
        }
    }

    /// Validate that dimensions are nonzero and fan_in <= input_neurons.
    pub fn validate(&self) -> Result<(), String> {
        if self.input_neurons == 0 {
            return Err("input_neurons must be > 0".into());
        }
        if self.hidden_neurons == 0 {
            return Err("hidden_neurons must be > 0".into());
        }
        if self.embedding_dim == 0 {
            return Err("embedding_dim must be > 0".into());
        }
        if self.fan_in == 0 || self.fan_in > self.input_neurons {
            return Err("fan_in must be in [1, input_neurons]".into());
        }
        if self.snn_steps == 0 {
            return Err("snn_steps must be > 0".into());
        }
        Ok(())
    }
}

impl Default for SnnBlockConfig {
    fn default() -> Self {
        Self::dim_4096()
    }
}

/// Gradients produced by back-propagating through the dense projection.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionGradients {
    pub d_weights: Vec<f32>,
    pub d_bias: Vec<f32>,
    pub d_features: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_4096() {
        let cfg = SnnBlockConfig::default();
        assert_eq!(cfg.input_neurons, 4096);
        assert_eq!(cfg.hidden_neurons, 4096);
        assert_eq!(cfg.embedding_dim, 4096);
    }

    #[test]
    fn spike_sample_active_count() {
        let sample = SpikeSample::new(vec![vec![0, 5], vec![5, 10], vec![0, 10, 20]]);
        assert_eq!(sample.active_neuron_count(), 4);
        assert_eq!(sample.steps(), 3);
    }

    #[test]
    fn config_rejects_invalid_fan_in() {
        let mut cfg = SnnBlockConfig::dim_4096();
        cfg.fan_in = 0;
        assert!(cfg.validate().is_err());

        cfg.fan_in = 5000;
        assert!(cfg.validate().is_err());
    }
}
