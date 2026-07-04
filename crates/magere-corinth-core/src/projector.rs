// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Spike-to-dense projection layer with learnable weights.

use crate::error::{Result, SnnError};
use crate::types::{ProjectionGradients, ProjectionMode};

const TEMPORAL_BINS: usize = 4;
const IZ_NEURONS: usize = 5;

#[allow(dead_code)]
fn feature_dim_for(snn_neurons: usize) -> usize {
    snn_neurons + (snn_neurons * TEMPORAL_BINS) + snn_neurons + IZ_NEURONS
}

/// Converts SNN spike activity into a dense embedding.
///
/// Internally this is a **learnable linear layer** `W ∈ ℝ^{embedding_dim × feature_dim}`
/// plus a bias `b ∈ ℝ^{embedding_dim}`. Weights are initialised with
/// Xavier-uniform values. Callers can update weights with an external optimizer
/// or use [`backward`](Self::backward) to obtain gradients.
///
/// When [`ProjectionMode::SpikingTernary`] is selected the projection uses GIF
/// membrane integration and fires ternary spikes (-1.0 / 0.0 / 1.0), producing
/// a sparse event-driven embedding. Membrane state persists across calls; call
/// [`reset_membrane`](Self::reset_membrane) to clear it (e.g. on sample
/// boundaries).
#[derive(Debug, Clone)]
pub struct SpikeToDenseProjector {
    mode: ProjectionMode,
    snn_neurons: usize,
    embedding_dim: usize,
    feature_dim: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
    rate_ema: Vec<f32>,
    ema_alpha: f32,
    // SpikingTernary state
    membrane: Vec<f32>,
    threshold: f32,
    decay: f32,
    // Cached feature vector from last forward for backward.
    last_features: Option<Vec<f32>>,
}

impl SpikeToDenseProjector {
    /// Create a projector with Xavier-uniform initialised weights.
    pub fn new(mode: ProjectionMode, snn_neurons: usize, embedding_dim: usize) -> Result<Self> {
        Self::with_bins(mode, snn_neurons, embedding_dim, TEMPORAL_BINS)
    }

    fn with_bins(
        mode: ProjectionMode,
        snn_neurons: usize,
        embedding_dim: usize,
        temporal_bins: usize,
    ) -> Result<Self> {
        if snn_neurons == 0 {
            return Err(SnnError::InvalidConfig("snn_neurons must be > 0".into()));
        }
        if embedding_dim == 0 {
            return Err(SnnError::InvalidConfig("embedding_dim must be > 0".into()));
        }

        let feature_dim = snn_neurons + (snn_neurons * temporal_bins) + snn_neurons + IZ_NEURONS;
        let fan_in = feature_dim as f32;
        let fan_out = embedding_dim as f32;
        let limit = (6.0_f32 / (fan_in + fan_out)).sqrt();
        const GOLDEN_RATIO_FRAC: f32 = 1.618_034;

        let mut weights = Vec::with_capacity(embedding_dim * feature_dim);
        for i in 0..(embedding_dim * feature_dim) {
            let t = ((i as f32 * GOLDEN_RATIO_FRAC) % 1.0) * 2.0 - 1.0;
            weights.push(t * limit);
        }

        Ok(Self {
            mode,
            snn_neurons,
            embedding_dim,
            feature_dim,
            weights,
            bias: vec![0.0; embedding_dim],
            rate_ema: vec![0.0; snn_neurons],
            ema_alpha: 0.1,
            membrane: vec![0.0; embedding_dim],
            threshold: 0.8,
            decay: 0.92,
            last_features: None,
        })
    }

    /// Project SNN spike activity into a dense embedding.
    pub fn project(
        &mut self,
        spike_train: &[Vec<usize>],
        potentials: &[f32],
        iz_potentials: &[f32],
    ) -> Result<Vec<f32>> {
        if potentials.len() < self.snn_neurons {
            return Err(SnnError::InputLengthMismatch {
                expected: self.snn_neurons,
                got: potentials.len(),
            });
        }

        let features = self.build_feature_vector(spike_train, potentials, iz_potentials);
        self.last_features = Some(features.clone());

        let embedding = match self.mode {
            ProjectionMode::SpikingTernary => self.spiking_linear_project(&features),
            _ => self.dense_linear_project(&features),
        };
        Ok(embedding)
    }

    /// Back-propagate through the dense projection.
    ///
    /// `d_output` is `dL/dembedding` and must have length `embedding_dim`.
    /// Returns gradients for weights, bias, and the feature vector.
    ///
    /// # Errors
    /// Returns [`SnnError::StateError`] if called before a forward pass.
    pub fn backward(&self, d_output: &[f32]) -> Result<ProjectionGradients> {
        if d_output.len() != self.embedding_dim {
            return Err(SnnError::InputLengthMismatch {
                expected: self.embedding_dim,
                got: d_output.len(),
            });
        }

        let features = self
            .last_features
            .as_ref()
            .ok_or_else(|| SnnError::StateError("backward called before project".into()))?;

        let mut d_weights = vec![0.0_f32; self.embedding_dim * self.feature_dim];
        let mut d_bias = vec![0.0_f32; self.embedding_dim];
        let mut d_features = vec![0.0_f32; self.feature_dim];

        for out_i in 0..self.embedding_dim {
            let d_y = d_output[out_i];
            d_bias[out_i] = d_y;

            let row_offset = out_i * self.feature_dim;
            for in_j in 0..self.feature_dim {
                let w = self.weights[row_offset + in_j];
                let x = features[in_j];
                d_weights[row_offset + in_j] = d_y * x;
                d_features[in_j] += d_y * w;
            }
        }

        Ok(ProjectionGradients {
            d_weights,
            d_bias,
            d_features,
        })
    }

    /// Apply a gradient update to weights and bias in-place.
    ///
    /// `learning_rate` is multiplied against the gradients. Callers that want
    /// more sophisticated optimizers should use [`weights`](Self::weights) and
    /// [`bias`](Self::bias) accessors.
    pub fn apply_gradients(
        &mut self,
        grads: &ProjectionGradients,
        learning_rate: f32,
    ) -> Result<()> {
        if grads.d_weights.len() != self.weights.len() {
            return Err(SnnError::InputLengthMismatch {
                expected: self.weights.len(),
                got: grads.d_weights.len(),
            });
        }
        if grads.d_bias.len() != self.bias.len() {
            return Err(SnnError::InputLengthMismatch {
                expected: self.bias.len(),
                got: grads.d_bias.len(),
            });
        }

        for (w, dw) in self.weights.iter_mut().zip(&grads.d_weights) {
            *w -= learning_rate * dw;
        }
        for (b, db) in self.bias.iter_mut().zip(&grads.d_bias) {
            *b -= learning_rate * db;
        }
        Ok(())
    }

    /// Replace the entire weight matrix.
    pub fn load_weights(&mut self, weights: &[f32]) -> Result<()> {
        let expected = self.embedding_dim * self.feature_dim;
        if weights.len() != expected {
            return Err(SnnError::InputLengthMismatch {
                expected,
                got: weights.len(),
            });
        }
        self.weights.copy_from_slice(weights);
        Ok(())
    }

    /// Replace the bias vector.
    pub fn load_bias(&mut self, bias: &[f32]) -> Result<()> {
        if bias.len() != self.embedding_dim {
            return Err(SnnError::InputLengthMismatch {
                expected: self.embedding_dim,
                got: bias.len(),
            });
        }
        self.bias.copy_from_slice(bias);
        Ok(())
    }

    /// Reset GIF membrane state.
    pub fn reset_membrane(&mut self) {
        self.membrane.fill(0.0);
    }

    /// Reset cached features.
    pub fn reset_cache(&mut self) {
        self.last_features = None;
    }

    /// Current projection mode.
    pub fn mode(&self) -> ProjectionMode {
        self.mode
    }

    /// Dimensionality constants: (feature_dim, embedding_dim).
    pub fn dims(&self) -> (usize, usize) {
        (self.feature_dim, self.embedding_dim)
    }

    /// Number of input SNN neurons.
    pub fn input_neurons(&self) -> usize {
        self.snn_neurons
    }

    /// Current weights (row-major: `W[out * feature_dim + in]`).
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Mutable weights for external optimizers.
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Current bias.
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Mutable bias for external optimizers.
    pub fn bias_mut(&mut self) -> &mut [f32] {
        &mut self.bias
    }

    /// Firing-rate exponential moving average snapshot.
    pub fn rate_ema(&self) -> &[f32] {
        &self.rate_ema
    }

    fn build_feature_vector(
        &mut self,
        spike_train: &[Vec<usize>],
        potentials: &[f32],
        iz_potentials: &[f32],
    ) -> Vec<f32> {
        let n_steps = spike_train.len().max(1) as f32;

        let mut rates = vec![0.0_f32; self.snn_neurons];
        for step in spike_train {
            for &idx in step {
                if idx < self.snn_neurons {
                    rates[idx] += 1.0;
                }
            }
        }
        for r in &mut rates {
            *r /= n_steps;
        }

        for (ema, rate) in self.rate_ema.iter_mut().zip(rates.iter()) {
            *ema = self.ema_alpha * *rate + (1.0 - self.ema_alpha) * *ema;
        }

        let bins = TEMPORAL_BINS;
        let mut hist = vec![0.0_f32; self.snn_neurons * bins];
        if !spike_train.is_empty() {
            let steps = spike_train.len();
            for (t, step) in spike_train.iter().enumerate() {
                let bin = ((t * bins) / steps).min(bins - 1);
                for &idx in step {
                    if idx < self.snn_neurons {
                        hist[idx * bins + bin] += 1.0;
                    }
                }
            }
            let total = n_steps / bins as f32;
            for h in &mut hist {
                *h /= total.max(1.0);
            }
        }

        let membrane: Vec<f32> = potentials[..self.snn_neurons]
            .iter()
            .map(|&v| v.clamp(0.0, 1.0))
            .collect();

        let iz: Vec<f32> = iz_potentials
            .iter()
            .take(IZ_NEURONS)
            .map(|&v| (v / 30.0).clamp(-1.0, 1.0))
            .chain(std::iter::repeat(0.0))
            .take(IZ_NEURONS)
            .collect();

        let mut features = Vec::with_capacity(self.feature_dim);
        match self.mode {
            ProjectionMode::RateSum => {
                features.extend_from_slice(&rates);
                features.extend_from_slice(&hist);
                features.extend_from_slice(&membrane);
                features.extend_from_slice(&iz);
            }
            ProjectionMode::TemporalHistogram => {
                let weighted_rates: Vec<f32> = rates.iter().map(|r| r * 0.3).collect();
                features.extend_from_slice(&weighted_rates);
                let weighted_hist: Vec<f32> = hist.iter().map(|h| h * 2.0).collect();
                features.extend_from_slice(&weighted_hist);
                features.extend_from_slice(&membrane);
                features.extend_from_slice(&iz);
            }
            ProjectionMode::MembraneSnapshot => {
                let membrane_primary: Vec<f32> = membrane.iter().map(|v| v * 2.0).collect();
                features.extend_from_slice(&rates);
                features.extend_from_slice(&hist);
                features.extend_from_slice(&membrane_primary);
                features.extend_from_slice(&iz);
            }
            ProjectionMode::SpikingTernary => {
                features.extend_from_slice(&rates);
                features.extend_from_slice(&hist);
                features.extend_from_slice(&membrane);
                features.extend_from_slice(&iz);
            }
        }

        features.resize(self.feature_dim, 0.0);
        features
    }

    fn dense_linear_project(&self, features: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.embedding_dim];
        for (out_i, out_slot) in out.iter_mut().enumerate() {
            let mut acc = self.bias[out_i];
            let row_offset = out_i * self.feature_dim;
            for (in_j, feature) in features.iter().take(self.feature_dim).enumerate() {
                acc += self.weights[row_offset + in_j] * *feature;
            }
            *out_slot = acc.tanh();
        }
        out
    }

    fn spiking_linear_project(&mut self, features: &[f32]) -> Vec<f32> {
        let mut spikes = vec![0.0_f32; self.embedding_dim];
        let activity_drive = features.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        for (out_i, spike) in spikes.iter_mut().enumerate() {
            let mut acc = self.bias[out_i];
            let row_offset = out_i * self.feature_dim;
            for (in_j, feature) in features.iter().take(self.feature_dim).enumerate() {
                acc += self.weights[row_offset + in_j] * *feature;
            }
            let drive = (acc.tanh() * 0.5 + activity_drive * 0.5).clamp(-1.0, 1.0);
            self.membrane[out_i] = self.membrane[out_i] * self.decay + drive * 0.35;
            if self.membrane[out_i] > self.threshold {
                *spike = 1.0;
                self.membrane[out_i] -= self.threshold;
            } else if self.membrane[out_i] < -self.threshold {
                *spike = -1.0;
                self.membrane[out_i] += self.threshold;
            }
        }
        spikes
    }
}

impl Default for SpikeToDenseProjector {
    fn default() -> Self {
        Self::new(ProjectionMode::RateSum, 4096, 4096).expect("default dimensions are valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_spike_train(n_steps: usize, neurons: usize) -> Vec<Vec<usize>> {
        (0..n_steps)
            .map(|t| vec![t % neurons, (t + 1) % neurons])
            .collect()
    }

    #[test]
    fn project_output_length() {
        let mut proj = SpikeToDenseProjector::new(ProjectionMode::RateSum, 128, 64).unwrap();
        let spikes = dummy_spike_train(20, 128);
        let potentials = vec![0.3; 128];
        let iz = vec![15.0; IZ_NEURONS];
        let emb = proj.project(&spikes, &potentials, &iz).unwrap();
        assert_eq!(emb.len(), 64);
    }

    #[test]
    fn project_values_bounded() {
        let mut proj =
            SpikeToDenseProjector::new(ProjectionMode::TemporalHistogram, 64, 32).unwrap();
        let spikes = dummy_spike_train(10, 64);
        let potentials = vec![0.5; 64];
        let iz = vec![30.0; IZ_NEURONS];
        let emb = proj.project(&spikes, &potentials, &iz).unwrap();
        assert!(emb.iter().all(|v| v.abs() <= 1.0 + 1e-6));
    }

    #[test]
    fn spiking_ternary_output() {
        let mut proj = SpikeToDenseProjector::new(ProjectionMode::SpikingTernary, 64, 32).unwrap();
        let spikes = dummy_spike_train(20, 64);
        let potentials = vec![0.3; 64];
        let iz = vec![15.0; IZ_NEURONS];
        let emb = proj.project(&spikes, &potentials, &iz).unwrap();
        assert_eq!(emb.len(), 32);
        assert!(
            emb.iter()
                .all(|&v| { (v - 1.0).abs() < 1e-6 || v.abs() < 1e-6 || (v + 1.0).abs() < 1e-6 })
        );
    }

    #[test]
    fn backward_requires_forward() {
        let proj = SpikeToDenseProjector::new(ProjectionMode::RateSum, 16, 8).unwrap();
        assert!(proj.backward(&[0.1; 8]).is_err());
    }

    #[test]
    fn backward_gradient_shapes() {
        let mut proj = SpikeToDenseProjector::new(ProjectionMode::RateSum, 16, 8).unwrap();
        let spikes = dummy_spike_train(4, 16);
        let potentials = vec![0.5; 16];
        let iz = vec![0.0; IZ_NEURONS];
        let _ = proj.project(&spikes, &potentials, &iz).unwrap();

        let grads = proj.backward(&[1.0; 8]).unwrap();
        assert_eq!(grads.d_weights.len(), proj.weights().len());
        assert_eq!(grads.d_bias.len(), 8);
        assert_eq!(grads.d_features.len(), proj.dims().0);
    }

    #[test]
    fn apply_gradients_changes_weights() {
        let mut proj = SpikeToDenseProjector::new(ProjectionMode::RateSum, 16, 8).unwrap();
        let spikes = dummy_spike_train(4, 16);
        let potentials = vec![0.5; 16];
        let iz = vec![0.0; IZ_NEURONS];
        let _ = proj.project(&spikes, &potentials, &iz).unwrap();

        let grads = proj.backward(&[1.0; 8]).unwrap();
        let before = proj.weights()[0];
        proj.apply_gradients(&grads, 0.01).unwrap();
        assert_ne!(proj.weights()[0], before);
    }

    #[test]
    fn custom_dims() {
        let neurons = 512;
        let emb = 256;
        let proj = SpikeToDenseProjector::new(ProjectionMode::RateSum, neurons, emb).unwrap();
        let (feature_dim, out_dim) = proj.dims();
        assert_eq!(out_dim, emb);
        assert_eq!(
            feature_dim,
            neurons + neurons * TEMPORAL_BINS + neurons + IZ_NEURONS
        );
    }
}
