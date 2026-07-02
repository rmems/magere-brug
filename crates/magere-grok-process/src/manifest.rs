// SPDX-License-Identifier: GPL-3.0-or-later
//! Manifest parsing — xai-dissect schema v1.

use crate::error::{GrokProcessError, Result};
use serde::{Deserialize, Serialize};

pub const MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const MANIFEST_NAME_CONVENTION_V1: &str = "xai-dissect-v1";

/// Root manifest document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DissectManifest {
    pub schema_version: u32,
    pub name_convention: String,
    pub model: ManifestModel,
    pub produced_by: ManifestProducedBy,
    pub defaults: ManifestDefaults,
    pub preserve: Vec<PreserveEntry>,
    pub fp16: Vec<Fp16Entry>,
    pub ternary_candidates: Vec<TernaryCandidate>,
}

/// Model metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestModel {
    pub family: String,
    pub name: String,
    pub total_params: u64,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_experts: usize,
    pub num_layers: usize,
    pub num_blocks: usize,
}

/// Provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestProducedBy {
    pub tool: String,
    pub version: String,
    pub timestamp: String,
}

/// Default precision tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestDefaults {
    pub router_precision: String,
    pub norm_precision: String,
    pub expert_precision: String,
}

/// Preserve-list entry (must not be ternarized).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreserveEntry {
    pub name: String,
    pub reason: String,
}

/// FP16-list entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fp16Entry {
    pub name: String,
    pub reason: String,
}

/// Ternary-candidate entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryCandidate {
    pub name: String,
    pub block: Option<usize>,
    pub slot: Option<usize>,
}

/// Parse manifest bytes and validate schema version.
pub fn parse_manifest_bytes(bytes: &[u8]) -> Result<DissectManifest> {
    let manifest: DissectManifest =
        serde_json::from_slice(bytes).map_err(|e| GrokProcessError::ManifestParse {
            path: "<bytes>".into(),
            source: e,
        })?;

    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(GrokProcessError::ManifestSchemaVersion {
            got: manifest.schema_version,
            expected: MANIFEST_SCHEMA_VERSION,
        });
    }

    if manifest.name_convention != MANIFEST_NAME_CONVENTION_V1 {
        return Err(GrokProcessError::ManifestNameConventionMismatch {
            got: manifest.name_convention.clone(),
            expected: MANIFEST_NAME_CONVENTION_V1.into(),
        });
    }

    Ok(manifest)
}

/// Load a manifest from a file path.
pub fn load_manifest(path: &std::path::Path) -> Result<DissectManifest> {
    let bytes = std::fs::read(path).map_err(|e| GrokProcessError::ManifestIo {
        path: path.display().to_string(),
        source: e,
    })?;
    parse_manifest_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> DissectManifest {
        DissectManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            name_convention: MANIFEST_NAME_CONVENTION_V1.into(),
            model: ManifestModel {
                family: "grok".into(),
                name: "Grok-1".into(),
                total_params: 314_000_000_000,
                vocab_size: 131_072,
                hidden_dim: 6144,
                num_experts: 8,
                num_layers: 64,
                num_blocks: 8,
            },
            produced_by: ManifestProducedBy {
                tool: "xai-dissect".into(),
                version: "0.1.0".into(),
                timestamp: "2024-01-01T00:00:00Z".into(),
            },
            defaults: ManifestDefaults {
                router_precision: "preserve".into(),
                norm_precision: "fp16".into(),
                expert_precision: "ternary_snn".into(),
            },
            preserve: vec![PreserveEntry {
                name: "router".into(),
                reason: "routing-critical".into(),
            }],
            fp16: vec![Fp16Entry {
                name: "norm".into(),
                reason: "numerical-stability".into(),
            }],
            ternary_candidates: vec![TernaryCandidate {
                name: "blk.0.weight".into(),
                block: Some(0),
                slot: None,
            }],
        }
    }

    #[test]
    fn parse_valid_manifest() {
        let manifest = sample_manifest();
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let parsed = parse_manifest_bytes(&bytes).unwrap();
        assert_eq!(parsed.model.hidden_dim, 6144);
    }

    #[test]
    fn reject_wrong_schema_version() {
        let mut manifest = sample_manifest();
        manifest.schema_version = 99;
        let bytes = serde_json::to_vec(&manifest).unwrap();
        assert!(parse_manifest_bytes(&bytes).is_err());
    }

    #[test]
    fn reject_wrong_name_convention() {
        let mut manifest = sample_manifest();
        manifest.name_convention = "legacy".into();
        let bytes = serde_json::to_vec(&manifest).unwrap();
        assert!(parse_manifest_bytes(&bytes).is_err());
    }
}
