use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model Quantization Manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub metadata: Metadata,
    pub model: ModelInfo,
    pub source_artifact: Artifact,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_artifact: Option<GeneratedArtifact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<Quantization>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_compatibility: Option<HashMap<String, BackendStatus>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saaq_experiment: Option<SAAQExperiment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benchmark_linkage: Option<BenchmarkLinkage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub schema_version: u32,
    pub created_at: String,
    pub manifest_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub slug: String,
    pub name: String,
    pub family: String,
    pub parameter_count: ParameterCount,
    pub architecture: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moe_layout: Option<MoELayout>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterCount {
    pub active: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoELayout {
    pub expert_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capacity_factor: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing_strategy: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub format: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<Checksum>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype_summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard_info: Option<ShardInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

/// Generated artifact with optional path for planned artifacts.
/// Prefer format `goz1` for hybrid ternary packs from magere-grok-process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedArtifact {
    pub format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// GOZ1 (or other pack) format version; aligns with GOZ1_VERSION when format is goz1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<Checksum>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype_summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard_info: Option<ShardInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_lineage: Option<SourceLineage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_summary: Option<TensorSummary>,
}

/// Lineage from a generated pack back to its source artifact or parent manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLineage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<Checksum>,
}

/// Tensor counts for a GOZ1 (or similar) pack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub f16_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ternary_count: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checksum {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub md5: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    pub shard_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard_size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard_paths: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quantization {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bits: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_dataset: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_config_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendStatus {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_types: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SAAQExperiment {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saaq_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing_entropy_metrics: Option<EntropyScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyScore {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_utilization: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entropy_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_balance_score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkLinkage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results_path: Option<String>,
}

impl Manifest {
    /// Load manifest from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Load manifest from file
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self::from_json(&content)?)
    }

    /// Validate required fields and supported format enums.
    pub fn validate(&self) -> Result<(), String> {
        const VALID_SOURCE_FORMATS: &[&str] = &["safetensors", "gguf", "hf_repo", "local_dir"];
        const VALID_GENERATED_FORMATS: &[&str] = &["goz1", "gguf", "ternary", "binary"];
        const VALID_QUANT_METHODS: &[&str] = &["gguf", "ternary", "binary", "saaq", "none"];
        const VALID_ARCHITECTURES: &[&str] = &["dense", "moe"];

        if self.metadata.schema_version < 1 {
            return Err("schema_version must be >= 1".to_string());
        }

        if self.metadata.created_at.is_empty() {
            return Err("metadata.created_at is required".to_string());
        }
        if chrono::DateTime::parse_from_rfc3339(&self.metadata.created_at).is_err() {
            return Err("metadata.created_at must be a valid RFC3339 timestamp".to_string());
        }

        if self.metadata.manifest_id.is_empty() {
            return Err("metadata.manifest_id is required".to_string());
        }

        if self.model.slug.is_empty() {
            return Err("model.slug is required".to_string());
        }

        if self.model.name.is_empty() {
            return Err("model.name is required".to_string());
        }

        if self.model.family.is_empty() {
            return Err("model.family is required".to_string());
        }
        if !VALID_ARCHITECTURES.contains(&self.model.architecture.as_str()) {
            return Err("model.architecture must be one of: dense, moe".to_string());
        }
        if self.model.architecture == "moe" && self.model.moe_layout.is_none() {
            return Err("model.moe_layout is required when architecture is moe".to_string());
        }

        if self.source_artifact.format.is_empty() {
            return Err("source_artifact.format is required".to_string());
        }
        if !VALID_SOURCE_FORMATS.contains(&self.source_artifact.format.as_str()) {
            return Err(
                "source_artifact.format must be one of: safetensors, gguf, hf_repo, local_dir"
                    .to_string(),
            );
        }

        if self.source_artifact.path.is_empty() {
            return Err("source_artifact.path is required".to_string());
        }

        if let Some(generated) = &self.generated_artifact {
            if generated.format.is_empty() {
                return Err(
                    "generated_artifact.format is required when generated_artifact is set"
                        .to_string(),
                );
            }
            if generated.format == "awq" || generated.format == "gptq" {
                return Err(format!(
                    "generated_artifact.format '{}' is not supported; use goz1 (or gguf/ternary/binary)",
                    generated.format
                ));
            }
            if !VALID_GENERATED_FORMATS.contains(&generated.format.as_str()) {
                return Err(
                    "generated_artifact.format must be one of: goz1, gguf, ternary, binary"
                        .to_string(),
                );
            }
            // Planned GOZ1 may omit path; if path is set, optionally check readability.
            if let Some(path) = &generated.path
                && !path.is_empty()
            {
                let p = std::path::Path::new(path);
                if p.exists() && !p.is_file() {
                    return Err(format!(
                        "generated_artifact.path exists but is not a file: {}",
                        path
                    ));
                }
            }
            if generated.format == "goz1"
                && let Some(version) = generated.version
                && version < 1
            {
                return Err("generated_artifact.version must be >= 1 for goz1".to_string());
            }
        }

        if let Some(quant) = &self.quantization
            && let Some(method) = &quant.method
        {
            if method == "awq" || method == "gptq" {
                return Err(format!(
                    "quantization.method '{}' is not supported; primary path is ternary/goz1/saaq",
                    method
                ));
            }
            if !VALID_QUANT_METHODS.contains(&method.as_str()) {
                return Err(
                    "quantization.method must be one of: gguf, ternary, binary, saaq, none"
                        .to_string(),
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_loads_valid_json() {
        let json = r#"{
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            }
        }"#;

        let manifest = Manifest::from_json(json);
        assert!(manifest.is_ok());
        let m = manifest.unwrap();
        assert_eq!(m.model.slug, "test_model");
    }

    #[test]
    fn test_required_fields_validate() {
        let json = r#"{
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            }
        }"#;

        let manifest = Manifest::from_json(json).unwrap();
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_missing_required_field_fails() {
        let json = r#"{
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            }
        }"#;

        let manifest = Manifest::from_json(json).unwrap();
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_checksum_field_shape() {
        let json = r#"{
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors",
                "checksum": {
                    "sha256": "abc123",
                    "md5": "def456"
                }
            }
        }"#;

        let manifest = Manifest::from_json(json).unwrap();
        assert!(manifest.source_artifact.checksum.is_some());
        let checksum = manifest.source_artifact.checksum.unwrap();
        assert_eq!(checksum.sha256, Some("abc123".to_string()));
        assert_eq!(checksum.md5, Some("def456".to_string()));
    }

    #[test]
    fn test_gguf_source_format_accepted() {
        let json = r#"{
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "gguf",
                "path": "/models/test.gguf"
            }
        }"#;

        let manifest = Manifest::from_json(json).unwrap();
        assert_eq!(manifest.source_artifact.format, "gguf");
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_safetensors_source_format_accepted() {
        let json = r#"{
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "moe",
                "moe_layout": {
                    "expert_count": 8
                }
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            }
        }"#;

        let manifest = Manifest::from_json(json).unwrap();
        assert_eq!(manifest.source_artifact.format, "safetensors");
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_generated_artifact_without_path() {
        let json = r#"{
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            },
            "generated_artifact": {
                "format": "ternary",
                "status": "planned"
            }
        }"#;

        let manifest = Manifest::from_json(json);
        assert!(manifest.is_ok());
        let m = manifest.unwrap();
        assert!(m.validate().is_ok());
        assert!(m.generated_artifact.is_some());
        let generated = m.generated_artifact.unwrap();
        assert_eq!(generated.format, "ternary");
        assert_eq!(generated.status, Some("planned".to_string()));
        assert!(generated.path.is_none());
    }

    #[test]
    fn test_goz1_generated_artifact_accepted() {
        let json = r#"{
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-goz1-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            },
            "generated_artifact": {
                "format": "goz1",
                "status": "success",
                "path": "/packs/test.goz1",
                "version": 1,
                "checksum": {"sha256": "abc123"},
                "size_bytes": 1024,
                "source_lineage": {
                    "manifest_id": "test-goz1-v1",
                    "path": "/models/test.safetensors"
                },
                "tensor_summary": {
                    "tensor_count": 10,
                    "f16_count": 2,
                    "ternary_count": 8
                }
            },
            "quantization": {
                "method": "ternary",
                "bits": 2
            }
        }"#;

        let m = Manifest::from_json(json).unwrap();
        assert!(m.validate().is_ok());
        let g = m.generated_artifact.unwrap();
        assert_eq!(g.format, "goz1");
        assert_eq!(g.version, Some(1));
        assert_eq!(g.tensor_summary.unwrap().tensor_count, Some(10));
    }

    #[test]
    fn test_planned_goz1_without_path_accepted() {
        let json = r#"{
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-goz1-planned"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "gguf",
                "path": "/models/test.gguf"
            },
            "generated_artifact": {
                "format": "goz1",
                "status": "planned",
                "version": 1
            },
            "quantization": {"method": "ternary", "bits": 2}
        }"#;

        let m = Manifest::from_json(json).unwrap();
        assert!(m.validate().is_ok());
    }

    #[test]
    fn test_awq_generated_format_rejected() {
        let json = r#"{
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-awq"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            },
            "generated_artifact": {
                "format": "awq",
                "status": "planned"
            }
        }"#;

        let m = Manifest::from_json(json).unwrap();
        let err = m.validate().unwrap_err();
        assert!(err.contains("awq") || err.contains("not supported"));
    }

    #[test]
    fn test_awq_quantization_method_rejected() {
        let json = r#"{
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-awq-method"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            },
            "quantization": {"method": "awq", "bits": 4}
        }"#;

        let m = Manifest::from_json(json).unwrap();
        let err = m.validate().unwrap_err();
        assert!(err.contains("awq") || err.contains("not supported"));
    }

    #[test]
    fn test_example_goz1_manifest_validates() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../manifests/examples/goz1-pack-example.json"
        );
        let m = Manifest::from_file(path).expect("load goz1 example");
        assert!(m.validate().is_ok());
        assert_eq!(m.generated_artifact.as_ref().unwrap().format, "goz1");
    }

    #[test]
    fn test_moe_requires_layout() {
        let json = r#"{
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "moe"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            }
        }"#;

        let manifest = Manifest::from_json(json).unwrap();
        assert!(manifest.validate().is_err());
    }
}
