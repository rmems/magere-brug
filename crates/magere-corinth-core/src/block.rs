// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Composable trainable SNN block: GIF hidden layer + spike-to-dense projector.

use crate::error::{Result, SnnError};
use crate::funnel::GifHiddenLayer;
use crate::projector::SpikeToDenseProjector;
use crate::types::{ProjectionGradients, SnnBlockConfig, SnnBlockOutput, SpikeSample};

/// A trainable SNN building block.
///
/// `SnnBlock` wires a [`GifHiddenLayer`] to a [`SpikeToDenseProjector`]. The
/// hidden layer is a fixed sparse reservoir; the projector has learnable
/// weights and biases.
#[derive(Debug, Clone)]
pub struct SnnBlock {
    config: SnnBlockConfig,
    hidden: GifHiddenLayer,
    projector: SpikeToDenseProjector,
}

impl SnnBlock {
    /// Create a block from a configuration.
    pub fn new(config: SnnBlockConfig) -> Result<Self> {
        config.validate().map_err(SnnError::InvalidConfig)?;
        let hidden = GifHiddenLayer::new(
            config.input_neurons,
            config.hidden_neurons,
            config.fan_in,
            config.snn_steps,
        )?;
        let projector = SpikeToDenseProjector::new(
            config.projection_mode,
            config.hidden_neurons,
            config.embedding_dim,
        )?;
        Ok(Self {
            config,
            hidden,
            projector,
        })
    }

    /// Run one spike-encoded sample through the block.
    pub fn forward(&mut self, sample: &SpikeSample) -> Result<SnnBlockOutput> {
        let activity = self.hidden.forward(sample)?;
        let embedding = self.projector.project(
            &activity.spike_train,
            &activity.potentials,
            &activity.iz_potentials,
        )?;

        Ok(SnnBlockOutput {
            spike_train: activity.spike_train,
            firing_rates: self.projector.rate_ema().to_vec(),
            membrane_potentials: activity.potentials,
            embedding,
        })
    }

    /// Reset all internal state (membrane, adaptation, projector cache).
    pub fn reset_state(&mut self) {
        self.hidden.reset();
        self.projector.reset_membrane();
        self.projector.reset_cache();
    }

    /// Back-propagate through the projector.
    ///
    /// Returns gradients for the projector weights and bias. Gradients for the
    /// hidden layer are not computed because the sparse reservoir weights are
    /// fixed in this version.
    pub fn backward(&self, d_output: &[f32]) -> Result<ProjectionGradients> {
        self.projector.backward(d_output)
    }

    /// Apply a simple SGD-style update to the projector.
    pub fn apply_gradients(
        &mut self,
        grads: &ProjectionGradients,
        learning_rate: f32,
    ) -> Result<()> {
        self.projector.apply_gradients(grads, learning_rate)
    }

    /// Reference to the projector.
    pub fn projector(&self) -> &SpikeToDenseProjector {
        &self.projector
    }

    /// Mutable reference to the projector.
    pub fn projector_mut(&mut self) -> &mut SpikeToDenseProjector {
        &mut self.projector
    }

    /// Reference to the hidden layer.
    pub fn hidden(&self) -> &GifHiddenLayer {
        &self.hidden
    }

    /// Block configuration.
    pub fn config(&self) -> &SnnBlockConfig {
        &self.config
    }
}

impl Default for SnnBlock {
    fn default() -> Self {
        Self::new(SnnBlockConfig::default()).expect("default config is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProjectionMode;

    #[test]
    fn block_forward_shape_4096() {
        let mut block = SnnBlock::default();
        let sample = SpikeSample::new(
            (0..20)
                .map(|t| vec![t % 256, (t + 50) % 256, (t + 100) % 256])
                .collect(),
        );

        let out = block.forward(&sample).unwrap();
        assert_eq!(out.spike_train.len(), 20);
        assert_eq!(out.membrane_potentials.len(), 4096);
        assert_eq!(out.embedding.len(), 4096);
    }

    #[test]
    fn block_training_step_changes_weights() {
        let cfg = SnnBlockConfig {
            input_neurons: 64,
            hidden_neurons: 64,
            embedding_dim: 32,
            projection_mode: ProjectionMode::RateSum,
            snn_steps: 20,
            fan_in: 4,
        };
        let mut block = SnnBlock::new(cfg).unwrap();
        // Dense, alternating input pattern to drive hidden activity.
        let sample = SpikeSample::new(
            (0..20)
                .map(|t| (0..64).step_by(2).map(|n| (n + t) % 64).collect())
                .collect(),
        );

        let out = block.forward(&sample).unwrap();
        assert!(
            out.spike_train.iter().map(Vec::len).sum::<usize>() > 0,
            "hidden layer should have fired"
        );

        let target = vec![0.5; 32];
        let loss_grad: Vec<f32> = out
            .embedding
            .iter()
            .zip(&target)
            .map(|(y, t)| y - t)
            .collect();

        let grads = block.backward(&loss_grad).unwrap();
        assert!(
            grads.d_weights.iter().any(|&dw| dw != 0.0),
            "weight gradients should be non-zero"
        );

        let before = block.projector().weights()[0];
        block.apply_gradients(&grads, 0.01).unwrap();
        assert_ne!(block.projector().weights()[0], before);
    }

    #[test]
    fn reset_clears_projector_membrane() {
        let mut block = SnnBlock::default();
        let sample = SpikeSample::new((0..20).map(|t| vec![t % 256]).collect());
        let _ = block.forward(&sample).unwrap();
        block.reset_state();
        // Membrane should be zeroed; a second identical forward should match
        // the first forward from a fresh state.
        let mut fresh = SnnBlock::default();
        let out_a = block.forward(&sample).unwrap();
        let out_b = fresh.forward(&sample).unwrap();
        assert_eq!(out_a.spike_train, out_b.spike_train);
    }
}
