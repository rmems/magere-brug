//! Recipe-driven ternary pack → GOZ1 runner (`magere pack-goz1`).
//!
//! Reads a `goz1_pack` / `ternary_pack` recipe (`schemas/recipe.schema.json`), maps its
//! `pack` block onto a [`QuantizeConfig`], runs `magere_grok_process::stream::run_quantize`,
//! writes the returned bytes to disk, re-reads the file and re-parses it as GOZ1, then records
//! the result as a `generated_artifact` in a freshly emitted model manifest which is finally
//! added to the artifact registry.
//!
//! # Skeleton caveat
//!
//! `magere-grok-process`'s end-to-end packer is an acknowledged **skeleton**: it walks the
//! dissect manifest's `ternary_candidates`, decides a precision tier per tensor, and emits a
//! 4-byte placeholder payload with a placeholder `[1, 1]` shape for each. It does **not** load
//! real tensor weights. This runner deliberately does not paper over that:
//!
//! * the CLI report states that every payload is a placeholder,
//! * the emitted `generated_artifact.status` is `planned` (never `success`), and
//! * the emitted `metadata.description` carries [`SKELETON_NOTICE`] verbatim.
//!
//! Loading real weights belongs to `magere-grok-process`, not here.

mod emit;
mod io;
mod paths;
mod recipe;
mod registry_lock;

pub use recipe::{PackRecipe, load_pack_recipe};

use crate::checksum;
use crate::manifest::Manifest;
use crate::registry::ArtifactRegistry;
use emit::{DissectInput, GeneratedManifestSpec, build_generated_manifest, format_outcome};
use io::{write_atomically, write_pack_durably};
use magere_grok_process::manifest::{DissectManifest, load_manifest};
use magere_grok_process::stream::run_quantize;
use magere_grok_process::types::QuantizeConfig;
use paths::{OutputLayout, reject_aliased_outputs, resolve_output_layout, resolve_source_manifest};
use recipe::{input_format_for, require_pack_config};
use registry_lock::{RegistryLock, load_registry};
use std::path::{Path, PathBuf};

/// Verbatim notice recorded on every artifact this runner produces.
pub const SKELETON_NOTICE: &str = "SKELETON PACK — magere-grok-process::stream::run_quantize does not load real tensor weights yet: every tensor payload is a 4-byte placeholder with shape [1, 1]. It also takes its QuantizeConfig as `_config` and discards it, so pack.input_dir, pack.input_format, pack.gif_threshold and pack.use_embedded_baseline are recorded but inert — only pack.dissect_manifest affects the emitted bytes. The file is a structurally valid GOZ1 shell, not a usable checkpoint, which is why generated_artifact.status is 'planned' rather than 'success'.";

/// Everything a caller (CLI or test) needs to know about a completed pack run.
#[derive(Debug, Clone)]
pub struct PackOutcome {
    pub recipe_id: String,
    pub pack_path: PathBuf,
    pub manifest_path: PathBuf,
    pub registry_path: PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
    pub tensor_count: u32,
    pub f16_count: u32,
    pub ternary_count: u32,
    pub manifest: Manifest,
    /// True when an existing registry entry for this slug was replaced.
    pub registry_replaced: bool,
    /// True when `input_dir` does not exist on disk. Harmless today — the skeleton packer
    /// never reads it — but it will matter once real tensor loading lands.
    pub input_dir_missing: bool,
}

/// Inputs gathered before any pack or registry write.
struct PreparedPack {
    recipe: PackRecipe,
    registry_file: PathBuf,
    _lock: RegistryLock,
    registry: ArtifactRegistry,
    source: Manifest,
    layout: OutputLayout,
    input_dir: String,
    quantize_config: QuantizeConfig,
    dissect: DissectManifest,
    dissect_input: DissectInput,
}

/// CLI entry point: run the recipe and render a human-readable report.
pub fn pack_goz1_command(
    recipe_path: &Path,
    registry_path: Option<&Path>,
    output_dir_override: Option<&Path>,
) -> Result<String, String> {
    let outcome = run_pack_recipe(recipe_path, registry_path, output_dir_override)?;
    Ok(format_outcome(&outcome))
}

/// Run a pack recipe end to end: quantize → write → verify → manifest → registry.
pub fn run_pack_recipe(
    recipe_path: &Path,
    registry_path: Option<&Path>,
    output_dir_override: Option<&Path>,
) -> Result<PackOutcome, String> {
    let prepared = prepare_pack_run(recipe_path, registry_path, output_dir_override)?;
    execute_pack_run(prepared)
}

fn prepare_pack_run(
    recipe_path: &Path,
    registry_path: Option<&Path>,
    output_dir_override: Option<&Path>,
) -> Result<PreparedPack, String> {
    let (recipe, pack_config, registry_file, lock, registry) =
        load_recipe_and_registry(recipe_path, registry_path)?;
    bind_inputs_and_outputs(
        recipe_path,
        recipe,
        pack_config,
        registry_file,
        lock,
        registry,
        output_dir_override,
    )
}

fn load_recipe_and_registry(
    recipe_path: &Path,
    registry_path: Option<&Path>,
) -> Result<
    (
        PackRecipe,
        recipe::PackConfig,
        PathBuf,
        RegistryLock,
        ArtifactRegistry,
    ),
    String,
> {
    let recipe = load_pack_recipe(recipe_path)?;
    let pack_config = require_pack_config(&recipe)?.clone();
    let registry_file = registry_path
        .unwrap_or_else(|| Path::new("registry.json"))
        .to_path_buf();
    let lock = RegistryLock::acquire(&registry_file)?;
    let registry = load_registry(&registry_file)?;
    Ok((recipe, pack_config, registry_file, lock, registry))
}

fn bind_inputs_and_outputs(
    recipe_path: &Path,
    recipe: PackRecipe,
    pack_config: recipe::PackConfig,
    registry_file: PathBuf,
    lock: RegistryLock,
    registry: ArtifactRegistry,
    output_dir_override: Option<&Path>,
) -> Result<PreparedPack, String> {
    let (source_path, source) = resolve_source_manifest(&recipe, &registry)?;
    let layout = resolve_output_layout(&recipe, &source, output_dir_override)?;
    let (quantize_config, input_dir) = quantize_config_from(&pack_config, &source, &layout)?;
    let (dissect, dissect_input) = load_dissect_input(&pack_config)?;
    reject_model_mismatch(
        &source,
        &dissect,
        pack_config.allow_model_mismatch.unwrap_or(false),
    )?;
    create_outputs_if_safe(
        recipe_path,
        &source_path,
        &registry_file,
        &layout,
        &dissect_input,
    )?;
    Ok(PreparedPack {
        recipe,
        registry_file,
        _lock: lock,
        registry,
        source,
        layout,
        input_dir,
        quantize_config,
        dissect,
        dissect_input,
    })
}

fn create_outputs_if_safe(
    recipe_path: &Path,
    source_path: &Path,
    registry_file: &Path,
    layout: &OutputLayout,
    dissect_input: &DissectInput,
) -> Result<(), String> {
    std::fs::create_dir_all(&layout.output_dir).map_err(|e| {
        format!(
            "Failed to create output dir '{}': {}",
            layout.output_dir.display(),
            e
        )
    })?;
    reject_aliased_outputs(
        recipe_path,
        source_path,
        &dissect_input.path,
        registry_file,
        &layout.pack_path,
        &layout.manifest_path,
    )
}

fn execute_pack_run(prepared: PreparedPack) -> Result<PackOutcome, String> {
    let PreparedPack {
        recipe,
        registry_file,
        _lock,
        mut registry,
        source,
        layout,
        input_dir,
        quantize_config,
        dissect,
        dissect_input,
        ..
    } = prepared;
    let bytes =
        run_quantize(&quantize_config, &dissect).map_err(|e| format!("GOZ1 pack failed: {}", e))?;
    write_pack_durably(&layout.pack_path, &bytes)?;
    let stats = io::verify_written_pack(&layout.pack_path, &bytes)?;
    let sha256 = checksum::compute_file_sha256(&layout.pack_path)
        .map_err(|e| format!("Failed to checksum '{}': {}", layout.pack_path.display(), e))?;
    let emitted = build_generated_manifest(GeneratedManifestSpec {
        recipe: &recipe,
        source: &source,
        manifest_id: layout.manifest_id.clone(),
        pack_path: &layout.pack_path,
        sha256: &sha256,
        stats,
        dissect_input: &dissect_input,
        resolved_input_dir: &input_dir,
    });
    write_emitted_manifest(&emitted, &layout.manifest_path)?;
    let registry_replaced = take_own_registry_entry(&mut registry, &emitted);
    register_emitted(&mut registry, &emitted, &layout)?;
    persist_registry(&registry, &registry_file)?;
    Ok(PackOutcome {
        recipe_id: recipe.recipe_id,
        pack_path: layout.pack_path,
        manifest_path: layout.manifest_path,
        registry_path: registry_file,
        sha256,
        size_bytes: stats.size_bytes,
        tensor_count: stats.tensor_count,
        f16_count: stats.f16_count,
        ternary_count: stats.ternary_count,
        manifest: emitted,
        registry_replaced,
        input_dir_missing: !Path::new(&input_dir).exists(),
    })
}

fn quantize_config_from(
    pack_config: &recipe::PackConfig,
    source: &Manifest,
    layout: &OutputLayout,
) -> Result<(QuantizeConfig, String), String> {
    let input_dir = pack_config
        .input_dir
        .clone()
        .unwrap_or_else(|| source.source_artifact.path.clone());
    let input_format = match pack_config.input_format {
        Some(format) => format,
        None => input_format_for(&source.source_artifact.format)?,
    };
    let gif_threshold = gif_threshold_from(pack_config)?;
    let dissect_manifest_path = PathBuf::from(&pack_config.dissect_manifest);
    Ok((
        QuantizeConfig {
            input_dir: input_dir.clone(),
            output_path: layout.pack_path.display().to_string(),
            gif_threshold,
            input_format,
            manifest_path: Some(dissect_manifest_path),
            use_embedded_baseline: pack_config.use_embedded_baseline.unwrap_or(false),
        },
        input_dir,
    ))
}

fn gif_threshold_from(pack_config: &recipe::PackConfig) -> Result<f32, String> {
    match pack_config.gif_threshold {
        Some(threshold) if threshold.is_finite() && (0.0..=1.0).contains(&threshold) => {
            Ok(threshold)
        }
        Some(threshold) => Err(format!(
            "pack.gif_threshold must be a finite number in [0.0, 1.0] (got {})",
            threshold
        )),
        None => Ok(QuantizeConfig::default().gif_threshold),
    }
}

fn load_dissect_input(
    pack_config: &recipe::PackConfig,
) -> Result<(DissectManifest, DissectInput), String> {
    let path = PathBuf::from(&pack_config.dissect_manifest);
    let dissect = load_manifest(&path).map_err(|e| {
        format!(
            "Failed to load dissect manifest '{}': {}",
            path.display(),
            e
        )
    })?;
    if dissect.ternary_candidates.is_empty() {
        return Err(format!(
            "dissect manifest '{}' lists no ternary_candidates; refusing to write an empty GOZ1 pack",
            path.display()
        ));
    }
    let sha256 = checksum::compute_file_sha256(&path).map_err(|e| {
        format!(
            "Failed to checksum dissect manifest '{}': {}",
            path.display(),
            e
        )
    })?;
    Ok((dissect, DissectInput { path, sha256 }))
}

fn reject_model_mismatch(
    source: &Manifest,
    dissect: &DissectManifest,
    allow_mismatch: bool,
) -> Result<(), String> {
    if allow_mismatch {
        return Ok(());
    }
    let source_family = source.model.family.trim();
    let dissect_family = dissect.model.family.trim();
    if source_family.eq_ignore_ascii_case(dissect_family) {
        return Ok(());
    }
    Err(format!(
        "source manifest family '{}' does not match dissect manifest family '{}' (source \
         model '{}', dissect model '{}'); pair compatible identities or set \
         pack.allow_model_mismatch=true",
        source_family, dissect_family, source.model.name, dissect.model.name
    ))
}

fn write_emitted_manifest(emitted: &Manifest, manifest_path: &Path) -> Result<(), String> {
    emitted
        .validate()
        .map_err(|e| format!("emitted manifest is invalid: {}", e))?;
    let serialized = serde_json::to_string_pretty(emitted)
        .map_err(|e| format!("Failed to serialize emitted manifest: {}", e))?;
    write_atomically(manifest_path, format!("{}\n", serialized).as_bytes()).map_err(|e| {
        format!(
            "Failed to write manifest '{}': {}",
            manifest_path.display(),
            e
        )
    })
}

fn take_own_registry_entry(registry: &mut ArtifactRegistry, emitted: &Manifest) -> bool {
    match registry.models.get(&emitted.model.slug) {
        Some(existing) if existing.manifest_id == emitted.metadata.manifest_id => {
            registry.models.remove(&emitted.model.slug);
            true
        }
        _ => false,
    }
}

fn register_emitted(
    registry: &mut ArtifactRegistry,
    emitted: &Manifest,
    layout: &OutputLayout,
) -> Result<(), String> {
    registry
        .register_at(emitted, Some(&layout.manifest_path))
        .map_err(|e| {
            format!(
                "{}\n  note: the pack and manifest were already written before this collision was \
                 detected:\n    {}\n    {}\n  they are left in place. The registry key is the \
                 generated slug '{}' (always `<source model slug>_goz1`); changing \
                 `outputs.manifest_id` cannot avoid this collision. Remove the conflicting \
                 registry entry or pack a different source model.",
                e,
                layout.pack_path.display(),
                layout.manifest_path.display(),
                emitted.model.slug
            )
        })
}

fn persist_registry(registry: &ArtifactRegistry, registry_file: &Path) -> Result<(), String> {
    let registry_json = registry
        .to_json_pretty()
        .map_err(|e| format!("Failed to serialize registry: {}", e))?;
    write_atomically(registry_file, registry_json.as_bytes())
        .map_err(|e| format!("Failed to write registry: {}", e))
}

#[cfg(test)]
mod tests;
