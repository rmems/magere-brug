// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Generalised Integrate-and-Fire (GIF) hidden layer for spike-encoded inputs.

use crate::error::{Result, SnnError};
use crate::types::SpikeSample;

/// Activity emitted by a forward pass through the hidden layer.
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenActivity {
    pub spike_train: Vec<Vec<usize>>,
    pub potentials: Vec<f32>,
    pub iz_potentials: Vec<f32>,
}

/// A sparse, deterministic GIF hidden layer.
///
/// Each hidden neuron receives input from a small fixed fan-in of input
/// neurons. Weights are deterministic (not learnable in this version) and
/// tuned so positive/negative input banks drive opposite response preferences.
///
/// The layer is stateful: membrane and adaptation persist across calls until
/// [`reset`](Self::reset) is called.
#[derive(Debug, Clone)]
pub struct GifHiddenLayer {
    input_neurons: usize,
    hidden_neurons: usize,
    fan_in: usize,
    snn_steps: usize,
    weight_indices: Vec<Vec<usize>>,
    weight_values: Vec<Vec<f32>>,
    membrane: Vec<f32>,
    adaptation: Vec<f32>,
    leak: f32,
    drive_scale: f32,
    threshold_base: f32,
    adaptation_scale: f32,
    adaptation_decay: f32,
    reset_ratio: f32,
}

impl GifHiddenLayer {
    /// Create a new hidden layer with the given dimensions.
    ///
    /// # Errors
    /// Returns [`SnnError::InvalidConfig`] if dimensions are zero or
    /// `fan_in > input_neurons`.
    pub fn new(
        input_neurons: usize,
        hidden_neurons: usize,
        fan_in: usize,
        snn_steps: usize,
    ) -> Result<Self> {
        if input_neurons == 0 {
            return Err(SnnError::InvalidConfig("input_neurons must be > 0".into()));
        }
        if hidden_neurons == 0 {
            return Err(SnnError::InvalidConfig("hidden_neurons must be > 0".into()));
        }
        if fan_in == 0 || fan_in > input_neurons {
            return Err(SnnError::InvalidConfig(
                "fan_in must be in [1, input_neurons]".into(),
            ));
        }
        if snn_steps == 0 {
            return Err(SnnError::InvalidConfig("snn_steps must be > 0".into()));
        }

        let (weight_indices, weight_values) =
            Self::build_weights(input_neurons, hidden_neurons, fan_in);

        Ok(Self {
            input_neurons,
            hidden_neurons,
            fan_in,
            snn_steps,
            weight_indices,
            weight_values,
            membrane: vec![0.0; hidden_neurons],
            adaptation: vec![0.0; hidden_neurons],
            leak: 0.92,
            drive_scale: 0.75,
            threshold_base: 0.65,
            adaptation_scale: 0.22,
            adaptation_decay: 0.94,
            reset_ratio: 0.35,
        })
    }

    /// Run one spike-encoded sample through the hidden layer.
    ///
    /// The sample's `potentials` field is ignored; the layer uses its own
    /// internal membrane state. Callers that need per-call initial potentials
    /// should call [`reset`](Self::reset) before `forward`.
    pub fn forward(&mut self, sample: &SpikeSample) -> Result<HiddenActivity> {
        if sample.spike_train.is_empty() {
            let potentials: Vec<f32> = self
                .membrane
                .iter()
                .map(|value| (value / (self.threshold_base * 2.0)).clamp(0.0, 1.0))
                .collect();

            return Ok(HiddenActivity {
                spike_train: vec![Vec::new(); self.snn_steps],
                potentials,
                iz_potentials: vec![0.0; 5],
            });
        }

        let mut spike_train: Vec<Vec<usize>> = Vec::with_capacity(sample.spike_train.len());
        let mut active = vec![false; self.input_neurons];

        for step in &sample.spike_train {
            active.fill(false);
            for &idx in step {
                if idx >= self.input_neurons {
                    return Err(SnnError::InputLengthMismatch {
                        expected: self.input_neurons,
                        got: idx + 1,
                    });
                }
                active[idx] = true;
            }

            let mut step_spikes = Vec::new();
            for hidden in 0..self.hidden_neurons {
                self.adaptation[hidden] *= self.adaptation_decay;

                let mut drive = 0.0_f32;
                let indices = &self.weight_indices[hidden];
                let values = &self.weight_values[hidden];
                for edge in 0..self.fan_in {
                    if active[indices[edge]] {
                        drive += values[edge];
                    }
                }

                self.membrane[hidden] = self.membrane[hidden] * self.leak
                    + drive * self.drive_scale
                    - self.adaptation[hidden] * 0.05;

                let threshold =
                    self.threshold_base + self.adaptation[hidden] * self.adaptation_scale;
                if self.membrane[hidden] >= threshold {
                    step_spikes.push(hidden);
                    self.membrane[hidden] -= threshold * self.reset_ratio;
                    self.adaptation[hidden] += 1.0;
                }
            }

            spike_train.push(step_spikes);
        }

        let potentials: Vec<f32> = self
            .membrane
            .iter()
            .map(|value| (value / (self.threshold_base * 2.0)).clamp(0.0, 1.0))
            .collect();

        Ok(HiddenActivity {
            spike_train,
            potentials,
            iz_potentials: vec![0.0; 5],
        })
    }

    /// Reset internal membrane and adaptation state.
    pub fn reset(&mut self) {
        self.membrane.fill(0.0);
        self.adaptation.fill(0.0);
    }

    /// Returns `true` if any membrane or adaptation value is non-zero.
    pub fn state_active(&self) -> bool {
        self.membrane.iter().any(|value| value.abs() > 1e-6)
            || self.adaptation.iter().any(|value| value.abs() > 1e-6)
    }

    /// Number of input neurons.
    pub fn input_neurons(&self) -> usize {
        self.input_neurons
    }

    /// Number of hidden neurons.
    pub fn hidden_neurons(&self) -> usize {
        self.hidden_neurons
    }

    fn build_weights(
        input_neurons: usize,
        hidden_neurons: usize,
        fan_in: usize,
    ) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let mut indices = Vec::with_capacity(hidden_neurons);
        let mut values = Vec::with_capacity(hidden_neurons);

        for hidden in 0..hidden_neurons {
            let tuned_negative = hidden % 2 == 1;
            let mut neuron_indices = Vec::with_capacity(fan_in);
            let mut neuron_values = Vec::with_capacity(fan_in);
            let mut cursor = (hidden * 11 + 3) % input_neurons;

            for edge in 0..fan_in {
                while neuron_indices.contains(&cursor) {
                    cursor = (cursor + 5) % input_neurons;
                }

                neuron_indices.push(cursor);

                let positive_bank = cursor % 4 < 2;
                let preference = if tuned_negative {
                    if positive_bank { -1.0 } else { 1.0 }
                } else if positive_bank {
                    1.0
                } else {
                    -1.0
                };
                let phase = ((hidden * 37 + edge * 19 + cursor * 13) % 97) as f32 / 96.0;
                neuron_values.push(preference * (0.35 + phase * 0.4));
                cursor = (cursor + 7 + hidden % 3) % input_neurons;
            }

            indices.push(neuron_indices);
            values.push(neuron_values);
        }

        (indices, values)
    }
}

impl Default for GifHiddenLayer {
    fn default() -> Self {
        Self::new(4096, 4096, 4, 20).expect("default dimensions are valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_rejects_zero_dimensions() {
        assert!(GifHiddenLayer::new(0, 4096, 4, 20).is_err());
        assert!(GifHiddenLayer::new(4096, 0, 4, 20).is_err());
        assert!(GifHiddenLayer::new(4096, 4096, 0, 20).is_err());
        assert!(GifHiddenLayer::new(4096, 4096, 4, 0).is_err());
    }

    #[test]
    fn layer_forward_output_shape() {
        let mut layer = GifHiddenLayer::new(128, 64, 4, 10).unwrap();
        let sample = SpikeSample::new((0..10).map(|t| vec![t % 128, (t + 1) % 128]).collect());

        let activity = layer.forward(&sample).unwrap();
        assert_eq!(activity.spike_train.len(), 10);
        assert_eq!(activity.potentials.len(), 64);
        assert_eq!(activity.iz_potentials.len(), 5);
    }

    #[test]
    fn layer_is_deterministic_from_fresh_state() {
        let sample = SpikeSample::new((0..12).map(|t| vec![t % 64]).collect());
        let mut a = GifHiddenLayer::new(64, 32, 4, 12).unwrap();
        let mut b = GifHiddenLayer::new(64, 32, 4, 12).unwrap();

        let out_a = a.forward(&sample).unwrap();
        let out_b = b.forward(&sample).unwrap();
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn reset_clears_state() {
        let sample = SpikeSample::new((0..20).map(|_| vec![0, 1, 2]).collect());
        let mut layer = GifHiddenLayer::new(64, 32, 4, 20).unwrap();
        let _ = layer.forward(&sample).unwrap();
        assert!(layer.state_active());
        layer.reset();
        assert!(!layer.state_active());
    }

    #[test]
    fn layer_rejects_out_of_bounds_spike_index() {
        let mut layer = GifHiddenLayer::new(8, 4, 2, 4).unwrap();
        let sample = SpikeSample::new(vec![vec![100]]);
        assert!(layer.forward(&sample).is_err());
    }
}
