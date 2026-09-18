use super::{DEFAULT_REGISTRY_PATH, Recipe, resolve::reference_is_manifest_path};
use crate::manifest::{GeneratedArtifact, Manifest};
use crate::registry::ArtifactRegistry;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

impl Recipe {
    pub(super) fn apply_register(
        &self,
        registry_override: Option<&Path>,
    ) -> Result<String, String> {
        let (manifest, manifest_path) = self.load_register_source()?;
        let registry_path = self.registry_destination(registry_override);
        write_registry(&manifest, &registry_path)?;
        let (emitted_manifest, emitted_handoff) =
            self.emit_registered_artifacts(&manifest, &manifest_path, &registry_path)?;
        Ok(format_register_report(
            self,
            &manifest,
            &manifest_path,
            &registry_path,
            &emitted_manifest,
            &emitted_handoff,
        ))
    }

    fn load_register_source(&self) -> Result<(Manifest, PathBuf), String> {
        let reference = self
            .source_manifest_ref()
            .ok_or_else(|| "register recipe requires inputs.source_manifest".to_string())?;

        if !reference_is_manifest_path(reference) {
            return Err(format!(
                "recipe '{}': inputs.source_manifest '{reference}' looks like a registry id; \
                 `magere recipe apply` needs a manifest path (a *.json file)",
                self.recipe_id
            ));
        }

        let should_register = self
            .outputs
            .as_ref()
            .is_none_or(|outputs| self.registers_output(outputs));
        if !should_register {
            return Err(format!(
                "recipe '{}': outputs.register is false, so there is nothing to apply",
                self.recipe_id
            ));
        }

        let manifest_path = self.resolve_reference(reference).ok_or_else(|| {
            format!("inputs.source_manifest '{reference}' could not be resolved to a file on disk")
        })?;
        let manifest = Manifest::from_file(&manifest_path)
            .map_err(|e| format!("failed to load manifest {}: {e}", manifest_path.display()))?;
        manifest.validate()?;
        Ok((manifest, manifest_path))
    }

    fn registry_destination(&self, registry_override: Option<&Path>) -> PathBuf {
        match registry_override {
            Some(path) => path.to_path_buf(),
            None => self
                .outputs
                .as_ref()
                .and_then(|outputs| outputs.registry_path.as_deref())
                .map_or_else(|| PathBuf::from(DEFAULT_REGISTRY_PATH), PathBuf::from),
        }
    }

    fn emit_registered_artifacts(
        &self,
        manifest: &Manifest,
        source_manifest_path: &Path,
        registry_path: &Path,
    ) -> Result<(PathBuf, PathBuf), String> {
        let base = emit_base_dir(self, registry_path);
        let manifests_dir = base.join("manifests");
        let handoff_dir = base.join("handoff");
        create_dir(&manifests_dir, "manifest")?;
        create_dir(&handoff_dir, "handoff")?;

        let emitted_manifest =
            manifests_dir.join(format!("{}.json", manifest.metadata.manifest_id));
        std::fs::copy(source_manifest_path, &emitted_manifest).map_err(|e| {
            format!(
                "failed to emit artifact manifest {}: {e}",
                emitted_manifest.display()
            )
        })?;

        let emitted_handoff = handoff_dir.join(format!("{}.json", manifest.metadata.manifest_id));
        let payload = combine_handoff_payload(self, manifest, &emitted_manifest);
        let serialized = serde_json::to_string_pretty(&payload)
            .map_err(|e| format!("failed to serialize combine-for-AI handoff: {e}"))?;
        std::fs::write(&emitted_handoff, serialized).map_err(|e| {
            format!(
                "failed to write combine-for-AI handoff {}: {e}",
                emitted_handoff.display()
            )
        })?;
        Ok((emitted_manifest, emitted_handoff))
    }
}

fn write_registry(manifest: &Manifest, registry_path: &Path) -> Result<(), String> {
    let mut registry = if registry_path.exists() {
        let content = std::fs::read_to_string(registry_path)
            .map_err(|e| format!("failed to read registry {}: {e}", registry_path.display()))?;
        ArtifactRegistry::from_json(&content)
            .map_err(|e| format!("failed to parse registry {}: {e}", registry_path.display()))?
    } else {
        ArtifactRegistry::new()
    };
    registry.register_or_update(manifest)?;
    let serialized = registry
        .to_json_pretty()
        .map_err(|e| format!("failed to serialize registry: {e}"))?;
    if let Some(parent) = registry_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create registry directory {}: {e}",
                parent.display()
            )
        })?;
    }
    std::fs::write(registry_path, serialized)
        .map_err(|e| format!("failed to write registry {}: {e}", registry_path.display()))
}

fn emit_base_dir(recipe: &Recipe, registry_path: &Path) -> PathBuf {
    recipe
        .outputs
        .as_ref()
        .and_then(|outputs| outputs.output_dir.as_deref())
        .map(PathBuf::from)
        .or_else(|| {
            registry_path.parent().and_then(|parent| {
                if parent.as_os_str().is_empty() {
                    None
                } else {
                    Some(parent.to_path_buf())
                }
            })
        })
        .unwrap_or_else(|| PathBuf::from("."))
}

fn create_dir(path: &Path, kind: &str) -> Result<(), String> {
    std::fs::create_dir_all(path)
        .map_err(|e| format!("failed to create {kind} directory {}: {e}", path.display()))
}

fn combine_handoff_payload(recipe: &Recipe, manifest: &Manifest, emitted_manifest: &Path) -> Value {
    let pipeline_id = recipe
        .handoff
        .as_ref()
        .and_then(|handoff| handoff.combine_for_ai.as_ref())
        .and_then(|target| target.pipeline_id.clone())
        .or_else(|| {
            manifest
                .benchmark_linkage
                .as_ref()
                .and_then(|linkage| linkage.pipeline_id.clone())
        });
    let kernel_types = recipe
        .handoff
        .as_ref()
        .and_then(|handoff| handoff.myelin_accelerator.as_ref())
        .and_then(|target| target.kernel_types.clone())
        .or_else(|| {
            manifest
                .backend_compatibility
                .as_ref()
                .and_then(|backends| backends.get("myelin_accelerator"))
                .and_then(|status| status.kernel_types.clone())
        });
    let checksum = manifest
        .source_artifact
        .checksum
        .as_ref()
        .and_then(|checksum| checksum.sha256.as_deref())
        .map(|sha256| format!("sha256:{sha256}"));

    let mut payload = json!({
        "schema": "magere-brug/combine-for-ai-handoff/1",
        "recipe_id": recipe.recipe_id,
        "manifest_ref": manifest.metadata.manifest_id,
        "model_slug": manifest.model.slug,
        "source_artifact": {
            "format": manifest.source_artifact.format,
            "path": manifest.source_artifact.path,
        },
        "emitted_manifest": emitted_manifest.display().to_string(),
        "benchmark_linkage": { "status": "ready" },
    });
    if let Some(checksum) = checksum {
        payload["source_artifact"]["checksum"] = Value::String(checksum);
    }
    if let Some(pipeline_id) = pipeline_id {
        payload["benchmark_linkage"]["pipeline_id"] = Value::String(pipeline_id);
    }
    if let Some(kernel_types) = kernel_types {
        payload["kernel_placeholders"] =
            Value::Array(kernel_types.into_iter().map(Value::String).collect());
    }
    payload
}

fn format_register_report(
    recipe: &Recipe,
    manifest: &Manifest,
    manifest_path: &Path,
    registry_path: &Path,
    emitted_manifest: &Path,
    emitted_handoff: &Path,
) -> String {
    let mut out = format!(
        "✓ Recipe '{}' ({}) applied\n  Manifest: {} ({})\n  Model: {} (slug: {}, family: {})\n  Source: {} {}\n",
        recipe.recipe_id,
        recipe.recipe_type,
        manifest_path.display(),
        manifest.metadata.manifest_id,
        manifest.model.name,
        manifest.model.slug,
        manifest.model.family,
        manifest.source_artifact.format,
        manifest.source_artifact.path
    );
    if let Some(generated) = &manifest.generated_artifact {
        append_generated_lines(&mut out, generated);
    }
    out.push_str(&format!(
        "  Registered to: {}\n  Emitted manifest: {}\n  combine-for-AI handoff: {}",
        registry_path.display(),
        emitted_manifest.display(),
        emitted_handoff.display()
    ));
    out
}

fn append_generated_lines(out: &mut String, generated: &GeneratedArtifact) {
    out.push_str(&format!(
        "  Generated artifact: {}{}{}\n",
        generated.format,
        generated
            .path
            .as_deref()
            .map(|path| format!(" {path}"))
            .unwrap_or_default(),
        generated
            .status
            .as_deref()
            .map(|status| format!(" (status: {status})"))
            .unwrap_or_default()
    ));
    if let Some(version) = generated.version {
        out.push_str(&format!("  Generated version: {version}\n"));
    }
    if let Some(sha256) = generated
        .checksum
        .as_ref()
        .and_then(|checksum| checksum.sha256.as_deref())
    {
        out.push_str(&format!("  Generated sha256: {sha256}\n"));
    }
    if let Some(lineage) = &generated.source_lineage {
        out.push_str(&format!(
            "  Generated lineage: {} <- {}\n",
            lineage.manifest_id.as_deref().unwrap_or("<unknown>"),
            lineage.path.as_deref().unwrap_or("<unknown>")
        ));
    }
}
