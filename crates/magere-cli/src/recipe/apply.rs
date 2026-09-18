use super::{
    DEFAULT_REGISTRY_PATH, Recipe,
    resolve::{is_filename_safe_id, reference_is_manifest_path},
};
use crate::manifest::{GeneratedArtifact, Manifest};
use crate::registry::ArtifactRegistry;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

struct RegisterSource {
    manifest: Manifest,
    path: PathBuf,
    bytes: Vec<u8>,
}

impl Recipe {
    pub(super) fn apply_register(
        &self,
        registry_override: Option<&Path>,
    ) -> Result<String, String> {
        let source = self.load_register_source()?;
        let registry_path = self.registry_destination(registry_override);
        let (emitted_manifest, emitted_handoff) =
            self.emit_registered_artifacts(&source, &registry_path)?;
        write_registry(&source.manifest, &registry_path)?;
        Ok(format_register_report(
            self,
            &source.manifest,
            &source.path,
            &registry_path,
            &emitted_manifest,
            &emitted_handoff,
        ))
    }

    fn load_register_source(&self) -> Result<RegisterSource, String> {
        let reference = require_manifest_path_ref(self)?;
        require_registration_requested(self)?;
        let path = self.resolve_reference(reference).ok_or_else(|| {
            format!("inputs.source_manifest '{reference}' could not be resolved to a file on disk")
        })?;
        read_validated_manifest(path)
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
        source: &RegisterSource,
        registry_path: &Path,
    ) -> Result<(PathBuf, PathBuf), String> {
        let base = emit_base_dir(self, registry_path);
        let manifests_dir = base.join("manifests");
        let handoff_dir = base.join("handoff");
        create_dir(&manifests_dir, "manifest")?;
        create_dir(&handoff_dir, "handoff")?;

        let file_name = emit_file_name(&source.manifest.metadata.manifest_id)?;
        let emitted_manifest = manifests_dir.join(&file_name);
        write_emitted_manifest(&source.path, &source.bytes, &emitted_manifest)?;

        let emitted_handoff = handoff_dir.join(&file_name);
        let payload = combine_handoff_payload(self, &source.manifest, &emitted_manifest);
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

fn read_validated_manifest(path: PathBuf) -> Result<RegisterSource, String> {
    let bytes = std::fs::read(&path)
        .map_err(|e| format!("failed to load manifest {}: {e}", path.display()))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| format!("failed to load manifest {}: {e}", path.display()))?;
    let manifest = Manifest::from_json(text)
        .map_err(|e| format!("failed to load manifest {}: {e}", path.display()))?;
    manifest.validate()?;
    Ok(RegisterSource {
        manifest,
        path,
        bytes,
    })
}

fn require_manifest_path_ref(recipe: &Recipe) -> Result<&str, String> {
    let reference = recipe
        .source_manifest_ref()
        .ok_or_else(|| "register recipe requires inputs.source_manifest".to_string())?;
    if !reference_is_manifest_path(reference) {
        return Err(format!(
            "recipe '{}': inputs.source_manifest '{reference}' looks like a registry id; \
             `magere recipe apply` needs a manifest path (a *.json file)",
            recipe.recipe_id
        ));
    }
    Ok(reference)
}

fn require_registration_requested(recipe: &Recipe) -> Result<(), String> {
    let should_register = recipe
        .outputs
        .as_ref()
        .is_none_or(|outputs| recipe.registers_output(outputs));
    if should_register {
        Ok(())
    } else {
        Err(format!(
            "recipe '{}': outputs.register is false, so there is nothing to apply",
            recipe.recipe_id
        ))
    }
}

fn emit_file_name(manifest_id: &str) -> Result<String, String> {
    if is_filename_safe_id(manifest_id) {
        Ok(format!("{manifest_id}.json"))
    } else {
        Err(format!(
            "metadata.manifest_id '{manifest_id}' must be a single filename-safe path component"
        ))
    }
}

fn write_emitted_manifest(source_path: &Path, bytes: &[u8], dest: &Path) -> Result<(), String> {
    if same_existing_file(source_path, dest) {
        return Ok(());
    }
    std::fs::write(dest, bytes)
        .map_err(|e| format!("failed to emit artifact manifest {}: {e}", dest.display()))
}

fn same_existing_file(left: &Path, right: &Path) -> bool {
    right.exists() && left.canonicalize().ok() == right.canonicalize().ok()
}

fn write_registry(manifest: &Manifest, registry_path: &Path) -> Result<(), String> {
    let mut registry = load_registry(registry_path)?;
    registry.register_or_update(manifest)?;
    let serialized = registry
        .to_json_pretty()
        .map_err(|e| format!("failed to serialize registry: {e}"))?;
    ensure_parent_dir(registry_path)?;
    std::fs::write(registry_path, serialized)
        .map_err(|e| format!("failed to write registry {}: {e}", registry_path.display()))
}

fn load_registry(registry_path: &Path) -> Result<ArtifactRegistry, String> {
    if !registry_path.exists() {
        return Ok(ArtifactRegistry::new());
    }
    let content = std::fs::read_to_string(registry_path)
        .map_err(|e| format!("failed to read registry {}: {e}", registry_path.display()))?;
    ArtifactRegistry::from_json(&content)
        .map_err(|e| format!("failed to parse registry {}: {e}", registry_path.display()))
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(parent).map_err(|e| {
        format!(
            "failed to create registry directory {}: {e}",
            parent.display()
        )
    })
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
    let status = benchmark_linkage_status(recipe);
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
        "benchmark_linkage": { "status": status },
    });
    if let Some(checksum) = source_checksum(manifest) {
        payload["source_artifact"]["checksum"] = Value::String(checksum);
    }
    if let Some(pipeline_id) = handoff_pipeline_id(recipe, manifest) {
        payload["benchmark_linkage"]["pipeline_id"] = Value::String(pipeline_id);
    }
    if let Some(kernel_types) = handoff_kernel_types(recipe, manifest) {
        payload["kernel_placeholders"] =
            Value::Array(kernel_types.into_iter().map(Value::String).collect());
    }
    payload
}

fn benchmark_linkage_status(recipe: &Recipe) -> String {
    match recipe
        .handoff
        .as_ref()
        .and_then(|handoff| handoff.combine_for_ai.as_ref())
    {
        Some(target) if target.enabled == Some(false) => target
            .status
            .clone()
            .unwrap_or_else(|| "placeholder".to_string()),
        Some(target) => target.status.clone().unwrap_or_else(|| "ready".to_string()),
        None => "ready".to_string(),
    }
}

fn handoff_pipeline_id(recipe: &Recipe, manifest: &Manifest) -> Option<String> {
    recipe
        .handoff
        .as_ref()
        .and_then(|handoff| handoff.combine_for_ai.as_ref())
        .and_then(|target| target.pipeline_id.clone())
        .or_else(|| {
            manifest
                .benchmark_linkage
                .as_ref()
                .and_then(|linkage| linkage.pipeline_id.clone())
        })
}

fn handoff_kernel_types(recipe: &Recipe, manifest: &Manifest) -> Option<Vec<String>> {
    recipe
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
        })
}

fn source_checksum(manifest: &Manifest) -> Option<String> {
    manifest
        .source_artifact
        .checksum
        .as_ref()
        .and_then(|checksum| checksum.sha256.as_deref())
        .map(|sha256| format!("sha256:{sha256}"))
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
