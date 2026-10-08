use crate::manifest::Manifest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
    /// Absolute or caller-supplied path of the registered manifest file.
    ///
    /// Used by `pack-goz1` when `inputs.source_manifest` is a registry slug or
    /// `manifest_id` rather than a filesystem path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_path: Option<PathBuf>,
}

impl ArtifactRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        ArtifactRegistry {
            version: 1,
            models: HashMap::new(),
        }
    }

    /// Register a model and remember the manifest file that described it.
    pub fn register_at(
        &mut self,
        manifest: &Manifest,
        manifest_path: Option<&Path>,
    ) -> Result<(), String> {
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
            manifest_path: manifest_path.map(Path::to_path_buf),
        };

        self.models.insert(manifest.model.slug.clone(), entry);
        Ok(())
    }

    /// Register a model, replacing an existing slug. Used by recipe apply so
    /// replaying a register recipe is idempotent.
    pub fn register_or_update(&mut self, manifest: &Manifest) -> Result<bool, String> {
        manifest.validate()?;

        let replaced = self.models.contains_key(&manifest.model.slug);
        let registered_at = self
            .models
            .get(&manifest.model.slug)
            .filter(|entry| entry.manifest_id == manifest.metadata.manifest_id)
            .map(|entry| entry.registered_at.clone())
            .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());
        let manifest_path = self
            .models
            .get(&manifest.model.slug)
            .and_then(|entry| entry.manifest_path.clone());
        let entry = RegistryEntry {
            slug: manifest.model.slug.clone(),
            manifest_id: manifest.metadata.manifest_id.clone(),
            family: manifest.model.family.clone(),
            status: "registered".to_string(),
            registered_at,
            notes: manifest.metadata.description.clone(),
            manifest_path,
        };

        self.models.insert(manifest.model.slug.clone(), entry);
        Ok(replaced)
    }

    /// Look up a model by slug
    #[allow(dead_code)]
    pub fn lookup(&self, slug: &str) -> Option<&RegistryEntry> {
        self.models.get(slug)
    }

    /// Look up a model by the manifest id recorded at registration.
    pub fn lookup_by_manifest_id(&self, manifest_id: &str) -> Option<&RegistryEntry> {
        self.models
            .values()
            .find(|entry| entry.manifest_id == manifest_id)
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

        let result = registry.register_at(&manifest, None);
        assert!(result.is_ok());
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_registry_lookup() {
        let mut registry = ArtifactRegistry::new();
        let manifest = create_test_manifest();
        registry.register_at(&manifest, None).unwrap();

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
        registry.register_at(&manifest1, None).unwrap();

        let list = registry.list_all();
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_registry_register_or_update_is_idempotent() {
        let mut registry = ArtifactRegistry::new();
        let manifest = create_test_manifest();

        assert!(!registry.register_or_update(&manifest).unwrap());
        let first_registered_at = registry.lookup("test_model").unwrap().registered_at.clone();
        assert!(registry.register_or_update(&manifest).unwrap());
        assert_eq!(registry.count(), 1);
        assert_eq!(
            registry.lookup("test_model").unwrap().manifest_id,
            "test-model-v1"
        );
        assert_eq!(
            registry.lookup("test_model").unwrap().registered_at,
            first_registered_at,
            "idempotent replay must preserve registered_at"
        );
    }
}
