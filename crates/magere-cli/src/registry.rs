use crate::manifest::Manifest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Producer recorded on entries written by `magere pack-goz1`.
pub const PACK_GOZ1_PRODUCER: &str = "pack-goz1";

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
    /// Absolute path of the registered manifest file.
    ///
    /// Used by `pack-goz1` when `inputs.source_manifest` is a registry slug or
    /// `manifest_id` rather than a filesystem path. Relative paths are
    /// canonicalized at registration so a later lookup does not depend on the
    /// caller's working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_path: Option<PathBuf>,
    /// Command that wrote this entry, when this CLI recorded one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer: Option<String>,
    /// `generated_artifact.status` at registration, when the manifest had one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_status: Option<String>,
    /// Recipe id recorded when a command wrote the entry.
    ///
    /// Replay identity is the manifest id, `producer`, and a non-success
    /// `generated_status`. This field is lineage, not part of that check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe_id: Option<String>,
}

/// How `admit_pack` says this slug may be written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackAdmission {
    Insert,
    ReplaceOwnPlanned,
}

/// Provenance stored with a pack-goz1 registry entry.
pub struct ProducedRegistration<'a> {
    pub manifest_path: &'a Path,
    pub producer: &'a str,
    pub generated_status: Option<&'a str>,
    pub recipe_id: Option<&'a str>,
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
        self.insert_entry(manifest, manifest_path, None)
    }

    /// Decide whether a pack may take `slug`, before any pack bytes are written.
    ///
    /// A replay replaces the entry only when it was written by `pack-goz1`,
    /// carries the same manifest id, and is still a planned skeleton.
    pub fn admit_pack(
        &self,
        slug: &str,
        manifest_id: &str,
        will_register: bool,
    ) -> Result<PackAdmission, String> {
        if let Some(owner) = self.conflicting_manifest_owner(manifest_id, slug) {
            return Err(format!(
                "outputs.manifest_id '{manifest_id}' is already registered to slug '{}'",
                owner.slug
            ));
        }
        match self.models.get(slug) {
            Some(existing) if will_register && own_planned_replay(existing, manifest_id) => {
                Ok(PackAdmission::ReplaceOwnPlanned)
            }
            Some(existing) => Err(format!(
                "model slug '{slug}' is already registered (manifest_id '{}', producer: {}). \
                 The registry key is always `<source model slug>_goz1`; changing \
                 `outputs.manifest_id` cannot avoid this collision.",
                existing.manifest_id,
                existing.producer.as_deref().unwrap_or("unspecified")
            )),
            None => Ok(PackAdmission::Insert),
        }
    }

    /// Another slug already owns this manifest id.
    pub fn conflicting_manifest_owner(
        &self,
        manifest_id: &str,
        slug: &str,
    ) -> Option<&RegistryEntry> {
        self.models
            .values()
            .find(|entry| entry.manifest_id == manifest_id && entry.slug != slug)
    }

    /// Insert a pack entry. `ReplaceOwnPlanned` removes the previous slug first.
    pub fn commit_pack(
        &mut self,
        manifest: &Manifest,
        produced: ProducedRegistration<'_>,
        admission: PackAdmission,
    ) -> Result<(), String> {
        if admission == PackAdmission::ReplaceOwnPlanned {
            self.models.remove(&manifest.model.slug);
        }
        self.insert_entry(manifest, Some(produced.manifest_path), Some(produced))
    }

    fn insert_entry(
        &mut self,
        manifest: &Manifest,
        manifest_path: Option<&Path>,
        produced: Option<ProducedRegistration<'_>>,
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
            manifest_path: stored_manifest_path(manifest_path)?,
            producer: produced
                .as_ref()
                .map(|produced| produced.producer.to_string()),
            generated_status: produced
                .as_ref()
                .and_then(|produced| produced.generated_status.map(str::to_string)),
            recipe_id: produced
                .as_ref()
                .and_then(|produced| produced.recipe_id.map(str::to_string)),
        };
        self.models.insert(manifest.model.slug.clone(), entry);
        Ok(())
    }

    /// Register a model, replacing an existing slug. Used by recipe apply so
    /// replaying a register recipe is idempotent.
    pub fn register_or_update(&mut self, manifest: &Manifest) -> Result<bool, String> {
        manifest.validate()?;

        let previous = self.models.get(&manifest.model.slug).cloned();
        let replaced = previous.is_some();
        let same_manifest = previous
            .as_ref()
            .is_some_and(|entry| entry.manifest_id == manifest.metadata.manifest_id);
        let registered_at = previous
            .as_ref()
            .filter(|_| same_manifest)
            .map(|entry| entry.registered_at.clone())
            .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());
        let manifest_path = previous
            .as_ref()
            .filter(|_| same_manifest)
            .and_then(|entry| entry.manifest_path.clone());
        let producer = previous
            .as_ref()
            .filter(|_| same_manifest)
            .and_then(|entry| entry.producer.clone());
        let generated_status = previous
            .as_ref()
            .filter(|_| same_manifest)
            .and_then(|entry| entry.generated_status.clone());
        let recipe_id = previous
            .as_ref()
            .filter(|_| same_manifest)
            .and_then(|entry| entry.recipe_id.clone());
        let entry = RegistryEntry {
            slug: manifest.model.slug.clone(),
            manifest_id: manifest.metadata.manifest_id.clone(),
            family: manifest.model.family.clone(),
            status: "registered".to_string(),
            registered_at,
            notes: manifest.metadata.description.clone(),
            manifest_path,
            producer,
            generated_status,
            recipe_id,
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

/// Store an absolute manifest path so later lookups survive a working-directory change.
pub fn canonical_manifest_path(path: &Path) -> Result<PathBuf, String> {
    std::fs::canonicalize(path).map_err(|e| {
        format!(
            "Failed to canonicalize manifest path '{}': {}",
            path.display(),
            e
        )
    })
}

fn own_planned_replay(existing: &RegistryEntry, manifest_id: &str) -> bool {
    existing.manifest_id == manifest_id
        && existing.producer.as_deref() == Some(PACK_GOZ1_PRODUCER)
        && existing.generated_status.as_deref() != Some("success")
}

fn stored_manifest_path(path: Option<&Path>) -> Result<Option<PathBuf>, String> {
    path.map(canonical_manifest_path).transpose()
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

    #[test]
    fn register_or_update_drops_the_manifest_path_when_the_id_changes() {
        let dir = tempfile::TempDir::new().unwrap();
        let manifest_path = dir.path().join("source.json");
        let mut manifest = create_test_manifest();
        std::fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let mut registry = ArtifactRegistry::new();
        registry
            .register_at(&manifest, Some(&manifest_path))
            .unwrap();
        manifest.metadata.manifest_id = "other-manifest".to_string();
        assert!(registry.register_or_update(&manifest).unwrap());
        assert!(
            registry
                .lookup("test_model")
                .unwrap()
                .manifest_path
                .is_none(),
            "a new manifest id must not keep the previous file path"
        );
    }

    #[test]
    fn register_at_stores_an_absolute_manifest_path() {
        let dir = tempfile::TempDir::new().unwrap();
        let manifest_path = dir.path().join("source.json");
        let manifest = create_test_manifest();
        std::fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let mut registry = ArtifactRegistry::new();
        registry
            .register_at(&manifest, Some(&manifest_path))
            .unwrap();
        let stored = registry
            .lookup("test_model")
            .unwrap()
            .manifest_path
            .clone()
            .unwrap();
        assert!(stored.is_absolute());
        assert!(stored.is_file());
        assert_eq!(stored, std::fs::canonicalize(&manifest_path).unwrap());
    }
}
