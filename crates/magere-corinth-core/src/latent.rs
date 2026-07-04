// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Training-time diagnostics for SNN blocks.

use crate::types::SnnBlockOutput;

/// Snapshot of training-relevant metrics from one forward pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SnnMetrics {
    pub total_spikes: usize,
    pub avg_pop_firing_rate_hz: f32,
    pub mean_membrane_potential: f32,
    pub membrane_dv_dt: f32,
    pub embedding_norm: f32,
    pub embedding_mean: f32,
    pub active_embedding_count: usize,
}

/// Compute metrics from an SNN block output.
///
/// `dt_seconds` is the duration of one simulation step. The default of `1e-3`
/// seconds (1 ms) matches a 1 kHz simulation clock.
pub fn compute_metrics(
    output: &SnnBlockOutput,
    hidden_neurons: usize,
    dt_seconds: f32,
) -> SnnMetrics {
    let dt_seconds = dt_seconds.max(1e-6);
    let steps = output.spike_train.len().max(1);
    let total_spikes = output.spike_train.iter().map(Vec::len).sum::<usize>();
    let avg_pop_firing_rate_hz =
        total_spikes as f32 / hidden_neurons.max(1) as f32 / (steps as f32 * dt_seconds);

    let mean_membrane_potential = mean(&output.membrane_potentials);
    let prev_mean_membrane = 0.0; // caller can track deltas across calls if desired
    let membrane_dv_dt = (mean_membrane_potential - prev_mean_membrane) / dt_seconds;

    let embedding_norm = l2_norm(&output.embedding);
    let embedding_mean = mean(&output.embedding);
    let active_embedding_count = output.embedding.iter().filter(|&&v| v.abs() > 1e-6).count();

    SnnMetrics {
        total_spikes,
        avg_pop_firing_rate_hz,
        mean_membrane_potential,
        membrane_dv_dt,
        embedding_norm,
        embedding_mean,
        active_embedding_count,
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_for_silent_output() {
        let out = SnnBlockOutput {
            spike_train: vec![Vec::new(); 10],
            firing_rates: vec![0.0; 64],
            membrane_potentials: vec![0.0; 64],
            embedding: vec![0.0; 32],
        };
        let m = compute_metrics(&out, 64, 1e-3);
        assert_eq!(m.total_spikes, 0);
        assert_eq!(m.active_embedding_count, 0);
        assert!(m.embedding_norm < 1e-6);
    }

    #[test]
    fn metrics_count_spikes_and_active_embeddings() {
        let mut embedding = vec![0.0; 8];
        embedding[0] = 0.5;
        embedding[3] = -0.25;

        let out = SnnBlockOutput {
            spike_train: vec![vec![0, 1], vec![2], Vec::new(), vec![0, 1, 2, 3]],
            firing_rates: vec![0.25; 8],
            membrane_potentials: vec![0.5; 8],
            embedding,
        };
        let m = compute_metrics(&out, 8, 1e-3);
        assert_eq!(m.total_spikes, 7);
        assert_eq!(m.active_embedding_count, 2);
        assert!(m.embedding_norm > 0.0);
    }
}
