use crate::manifest::Manifest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Artifact Registry - tracks all registered models and their manifests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRegistry {
    pub version: u32,
    pub models: HashMap<String, RegistryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub slug: String,
    pub manifest_id: String,
    pub family: String,
    pub status: String,
    pub registered_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl ArtifactRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        ArtifactRegistry {
            version: 1,
            models: HashMap::new(),
        }
    }

    /// Register a model from its manifest
    pub fn register(&mut self, manifest: &Manifest) -> Result<(), String> {
        manifest.validate()?;

        if self.models.contains_key(&manifest.model.slug) {
            return Err(format!(
                "model slug '{}' is already registered",
                manifest.model.slug
            ));
        }

        let entry = RegistryEntry {
            slug: manifest.model.slug.clone(),
            manifest_id: manifest.metadata.manifest_id.clone(),
            family: manifest.model.family.clone(),
            status: "registered".to_string(),
            registered_at: chrono::Utc::now().to_rfc3339(),
            notes: manifest.metadata.description.clone(),
        };

        self.models.insert(manifest.model.slug.clone(), entry);
        Ok(())
    }

    /// Look up a model by slug
    #[allow(dead_code)]
    pub fn lookup(&self, slug: &str) -> Option<&RegistryEntry> {
        self.models.get(slug)
    }

    /// List all registered models
    #[allow(dead_code)]
    pub fn list_all(&self) -> Vec<&RegistryEntry> {
        self.models.values().collect()
    }

    /// Count registered models
    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        self.models.len()
    }

    /// Serialize to JSON
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for ArtifactRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_manifest() -> Manifest {
        use crate::manifest::*;

        Manifest {
            metadata: Metadata {
                schema_version: 1,
                created_at: "2026-05-26T00:00:00Z".to_string(),
                manifest_id: "test-model-v1".to_string(),
                description: Some("Test model".to_string()),
            },
            model: ModelInfo {
                slug: "test_model".to_string(),
                name: "Test Model".to_string(),
                family: "test".to_string(),
                parameter_count: ParameterCount {
                    active: 1000000,
                    total: Some(1000000),
                },
                architecture: "dense".to_string(),
                moe_layout: None,
            },
            source_artifact: Artifact {
                format: "safetensors".to_string(),
                path: "/models/test.safetensors".to_string(),
                source_url: None,
                checksum: None,
                dtype_summary: None,
                size_bytes: None,
                shard_info: None,
                timestamp: None,
            },
            generated_artifact: None,
            quantization: None,
            backend_compatibility: None,
            saaq_experiment: None,
            benchmark_linkage: None,
        }
    }

    #[test]
    fn test_registry_new() {
        let registry = ArtifactRegistry::new();
        assert_eq!(registry.version, 1);
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_registry_register() {
        let mut registry = ArtifactRegistry::new();
        let manifest = create_test_manifest();

        let result = registry.register(&manifest);
        assert!(result.is_ok());
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_registry_lookup() {
        let mut registry = ArtifactRegistry::new();
        let manifest = create_test_manifest();
        registry.register(&manifest).unwrap();

        let entry = registry.lookup("test_model");
        assert!(entry.is_some());
        let e = entry.unwrap();
        assert_eq!(e.slug, "test_model");
        assert_eq!(e.family, "test");
    }

    #[test]
    fn test_registry_lookup_missing() {
        let registry = ArtifactRegistry::new();
        let entry = registry.lookup("nonexistent");
        assert!(entry.is_none());
    }

    #[test]
    fn test_registry_list_all() {
        let mut registry = ArtifactRegistry::new();
        let manifest1 = create_test_manifest();
        registry.register(&manifest1).unwrap();

        let list = registry.list_all();
        assert_eq!(list.len(), 1);
    }
}
