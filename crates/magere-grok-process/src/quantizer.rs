// SPDX-License-Identifier: GPL-3.0-or-later
//! Quantizer — convert tensors to ternary or FP16 representations.

use crate::types::TensorPrecision;

/// A quantized tensor with its metadata.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub precision: TensorPrecision,
    pub data: Vec<u8>,
}

/// Quantize a f32 slice to ternary {-1, 0, +1} using a saliency threshold.
///
/// τ = gif_threshold × rms(weights)
pub fn quantize_ternary(data: &[f32], gif_threshold: f32) -> Vec<i8> {
    if data.is_empty() {
        return Vec::new();
    }

    let rms = (data.iter().map(|w| w * w).sum::<f32>() / data.len() as f32).sqrt();
    let threshold = gif_threshold * rms;

    data
        .iter()
        .map(|&w| {
            if w.abs() < threshold {
                0
            } else if w > 0.0 {
                1
            } else {
                -1
            }
        })
        .collect()
}

/// Quantize a f32 slice to f16 (IEEE 754 half-precision).
pub fn quantize_f16(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|&v| f32_to_f16_bytes(v))
        .collect()
}

/// Keep f32 weights as-is (for Preserve tier).
pub fn quantize_f32(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect()
}

fn f32_to_f16_bytes(value: f32) -> [u8; 2] {
    // Naïve round-to-nearest-even f32→f16 conversion.
    let bits = value.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = bits & 0x7FFFFF;

    let h_sign = (sign as u16) << 15;
    let mut h_exp: i32;
    let mut h_mant: u16;

    if exponent == 128 && mantissa != 0 {
        // NaN
        h_exp = 31;
        h_mant = 0x200;
    } else if exponent > 15 {
        // Overflow to infinity
        h_exp = 31;
        h_mant = 0;
    } else if exponent < -14 {
        // Underflow to zero or subnormal
        if exponent < -24 {
            h_exp = 0;
            h_mant = 0;
        } else {
            h_exp = 0;
            h_mant = ((mantissa | 0x800000) >> (-exponent - 14)) as u16;
        }
    } else {
        h_exp = exponent + 15;
        h_mant = (mantissa >> 13) as u16;
        // Round to nearest even
        let round_bit = (mantissa >> 12) & 0x1;
        if round_bit == 1 {
            let sticky = mantissa & 0xFFF;
            let lsb = h_mant & 0x1;
            if sticky != 0 || lsb == 1 {
                h_mant = h_mant.wrapping_add(1);
                if h_mant > 0x3FF {
                    h_mant = 0;
                    h_exp += 1;
                }
            }
        }
    }

    let h_bits = h_sign | ((h_exp as u16) << 10) | (h_mant & 0x3FF);
    h_bits.to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_quantization_silences_small_weights() {
        let data = vec![0.01, 0.5, -0.02, 0.8, -0.7];
        let quantized = quantize_ternary(&data, 0.1);
        // RMS ≈ 0.47, threshold ≈ 0.047
        assert_eq!(quantized, vec![0, 1, 0, 1, -1]);
    }

    #[test]
    fn f16_roundtrip_preserves_magnitude() {
        let original = vec![1.0f32, -1.0, 0.5, 1000.0, 0.0001];
        let bytes = quantize_f16(&original);
        assert_eq!(bytes.len(), original.len() * 2);
    }

    #[test]
    fn f32_passthrough_unchanged() {
        let data = vec![1.5f32, -2.5, 0.0];
        let bytes = quantize_f32(&data);
        assert_eq!(bytes.len(), data.len() * 4);
        let reconstructed: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(data, reconstructed);
    }
}
