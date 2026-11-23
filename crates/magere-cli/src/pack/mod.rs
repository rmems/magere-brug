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

pub use recipe::load_pack_recipe;

use crate::atomic_write::write_bytes_atomically;
use crate::checksum;
use crate::manifest::Manifest;
use crate::recipe::Recipe;
use crate::registry::{ArtifactRegistry, PACK_GOZ1_PRODUCER, PackAdmission, ProducedRegistration};
use crate::registry_io::{self, RegistryLock};
use emit::{DissectInput, GeneratedManifestSpec, build_generated_manifest, format_outcome};
use magere_grok_process::manifest::{DissectManifest, load_manifest};
use magere_grok_process::stream::run_quantize;
use magere_grok_process::types::QuantizeConfig;
use paths::{OutputLayout, reject_aliased_outputs, resolve_output_layout, resolve_source_manifest};
use recipe::{PackConfig, input_format_for, require_pack_config};
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
    /// True when the recipe declared `calibration`. The skeleton packer does not apply it.
    pub calibration_declared: bool,
    /// True when the registry was updated. `outputs.register: false` leaves it unchanged.
    pub registered: bool,
}

/// The registry path as spelled plus the destination it resolves to.
struct RegistryPaths {
    /// As spelled by the recipe or CLI — for reporting.
    report: PathBuf,
    /// Resolved destination — lock, load, collision checks, store.
    dest: PathBuf,
}

/// Inputs gathered before any pack or registry write.
struct PreparedPack {
    recipe: Recipe,
    registry_paths: RegistryPaths,
    _lock: RegistryLock,
    _output_lock: RegistryLock,
    registry: ArtifactRegistry,
    source: Manifest,
    layout: OutputLayout,
    input_dir: String,
    quantize_config: QuantizeConfig,
    dissect: DissectManifest,
    dissect_input: DissectInput,
    admission: PackAdmission,
    will_register: bool,
    calibration_declared: bool,
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
    let (recipe, pack_config, registry_paths, lock, registry) =
        load_recipe_and_registry(recipe_path, registry_path)?;
    bind_inputs_and_outputs(
        recipe_path,
        recipe,
        pack_config,
        registry_paths,
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
        Recipe,
        PackConfig,
        RegistryPaths,
        RegistryLock,
        ArtifactRegistry,
    ),
    String,
> {
    let recipe = load_pack_recipe(recipe_path)?;
    let pack_config = require_pack_config(&recipe)?;
    let registry_file = registry_path.map(Path::to_path_buf).unwrap_or_else(|| {
        recipe
            .outputs
            .as_ref()
            .and_then(|outputs| outputs.registry_path.as_deref())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("registry.json"))
    });
    // Resolve the registry destination before any output is written: a
    // dangling symlink must name its target for the lock, the load, the
    // collision checks, and the store alike — otherwise the store could
    // follow the link and overwrite an emitted artifact. The spelled path is
    // kept for reporting; every filesystem operation uses the destination.
    let registry_paths = RegistryPaths {
        dest: crate::registry_io::registry_identity(&registry_file),
        report: registry_file,
    };
    let lock = RegistryLock::acquire(&registry_paths.dest)?;
    let registry = registry_io::load_registry(&registry_paths.dest)?;
    Ok((recipe, pack_config, registry_paths, lock, registry))
}

fn bind_inputs_and_outputs(
    recipe_path: &Path,
    recipe: Recipe,
    pack_config: PackConfig,
    registry_paths: RegistryPaths,
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
    let will_register = recipe
        .outputs
        .as_ref()
        .is_none_or(|outputs| recipe.registers_output(outputs));
    let slug = format!("{}_goz1", source.model.slug);
    let admission = admit_registry_write(
        &registry,
        &slug,
        &layout.manifest_id,
        &layout.pack_path,
        &layout.manifest_path,
        will_register,
    )?;
    paths::reject_output_aliases(
        &registry,
        &slug,
        &layout.pack_path,
        &layout.manifest_path,
        will_register,
    )?;
    recipe.check_pack_source(&source, &input_dir)?;
    // outputs.artifact_path asserts the recipe's own derived destination; an
    // explicit --output-dir makes that assertion meaningless for this run.
    if output_dir_override.is_none() {
        check_artifact_path(&recipe, &layout)?;
    }
    let calibration_declared = recipe.calibration.is_some();
    create_outputs_if_safe(
        recipe_path,
        &source_path,
        &registry_paths,
        &layout,
        &dissect_input,
    )?;
    // The registry lock cannot coordinate runs that use different registry
    // files, so the output directory gets its own lock: two packs writing the
    // same destination serialize on it regardless of which registry they pick.
    let output_lock = acquire_output_lock(&registry_paths.dest, &layout.output_dir)?;
    Ok(PreparedPack {
        recipe,
        registry_paths,
        _lock: lock,
        _output_lock: output_lock,
        registry,
        source,
        layout,
        input_dir,
        quantize_config,
        dissect,
        dissect_input,
        admission,
        will_register,
        calibration_declared,
    })
}

fn create_outputs_if_safe(
    recipe_path: &Path,
    source_path: &Path,
    registry_paths: &RegistryPaths,
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
        &[&registry_paths.report, &registry_paths.dest],
        &layout.pack_path,
        &layout.manifest_path,
    )
}

/// Lock `<output-dir>/.pack-goz1` for the duration of the pack run. A registry
/// path that resolves to the same lock file — e.g. `--registry
/// <output-dir>/.pack-goz1` — would make this process wait on a lock it
/// already holds, so it is refused before the second acquire.
fn acquire_output_lock(registry_file: &Path, output_dir: &Path) -> Result<RegistryLock, String> {
    let output_lock_base = output_dir.join(".pack-goz1");
    let output_lock_path = crate::registry_io::registry_lock_path(&output_lock_base);
    if crate::registry_io::registry_lock_path(registry_file) == output_lock_path {
        return Err(format!(
            "registry path '{}' and the pack output lock would share '{}'; \
             choose a different registry path",
            registry_file.display(),
            output_lock_path.display()
        ));
    }
    RegistryLock::acquire(&output_lock_base)
}

fn execute_pack_run(prepared: PreparedPack) -> Result<PackOutcome, String> {
    let PreparedPack {
        recipe,
        registry_paths,
        _lock,
        _output_lock,
        mut registry,
        source,
        layout,
        input_dir,
        quantize_config,
        dissect,
        dissect_input,
        admission,
        will_register,
        calibration_declared,
    } = prepared;
    let bytes =
        run_quantize(&quantize_config, &dissect).map_err(|e| format!("GOZ1 pack failed: {}", e))?;
    let stats = io::validate_pack_bytes(&bytes)?;
    let sha256 = checksum::compute_bytes_sha256(&bytes);
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
    emitted
        .validate()
        .map_err(|e| format!("emitted manifest is invalid: {e}"))?;
    if will_register {
        registry.rehearse_pack_commit(&emitted, admission)?;
    }
    // A replay can overwrite files an existing registry entry still names, so
    // snapshot them before the first write: any failure below must restore the
    // previous outputs rather than leave dangling registry references.
    let guard = OutputGuard::snapshot(&layout)?;
    let committed = (|| -> Result<(io::PackStats, bool), String> {
        write_bytes_atomically(&layout.pack_path, &bytes)?;
        let stats = verify_pack_and_emit_manifest(&layout, &bytes, &emitted)?;
        let registry_replaced = if will_register {
            register_and_persist(
                &mut registry,
                &emitted,
                &layout,
                &recipe.recipe_id,
                admission,
                &registry_paths.dest,
            )?
        } else {
            false
        };
        Ok((stats, registry_replaced))
    })();
    let (stats, registry_replaced) = match committed {
        Ok(pair) => pair,
        Err(err) => {
            let rollback_failures = guard.restore();
            return Err(match rollback_failures.is_empty() {
                true => err,
                false => format!(
                    "{err}\n  note: rollback was incomplete: {}",
                    rollback_failures.join("; ")
                ),
            });
        }
    };
    drop(guard);
    Ok(PackOutcome {
        recipe_id: recipe.recipe_id,
        pack_path: layout.pack_path,
        manifest_path: layout.manifest_path,
        registry_path: registry_paths.report,
        sha256,
        size_bytes: stats.size_bytes,
        tensor_count: stats.tensor_count,
        f16_count: stats.f16_count,
        ternary_count: stats.ternary_count,
        manifest: emitted,
        registry_replaced,
        input_dir_missing: !Path::new(&input_dir).exists(),
        calibration_declared,
        registered: will_register,
    })
}

fn admit_registry_write(
    registry: &ArtifactRegistry,
    slug: &str,
    manifest_id: &str,
    pack_path: &Path,
    manifest_path: &Path,
    will_register: bool,
) -> Result<PackAdmission, String> {
    registry
        .admit_pack(slug, manifest_id, will_register)
        .map_err(|reason| {
            format!(
                "{reason}\n  Refusing to write:\n    {}\n    {}\n  Nothing was written.",
                pack_path.display(),
                manifest_path.display()
            )
        })
}

fn check_artifact_path(recipe: &Recipe, layout: &OutputLayout) -> Result<(), String> {
    let Some(artifact_path) = recipe
        .outputs
        .as_ref()
        .and_then(|outputs| outputs.artifact_path.as_deref())
    else {
        return Ok(());
    };
    paths::reject_parent_dir_components(Path::new(artifact_path), "outputs.artifact_path")?;
    let declared = paths::resolve_planned_path(Path::new(artifact_path))?;
    let derived = paths::resolve_planned_path(&layout.pack_path)?;
    if declared != derived {
        return Err(format!(
            "outputs.artifact_path '{}' does not match the pack path derived from \
             output_dir and manifest_id ('{}')",
            artifact_path,
            layout.pack_path.display()
        ));
    }
    Ok(())
}

fn quantize_config_from(
    pack_config: &PackConfig,
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

fn gif_threshold_from(pack_config: &PackConfig) -> Result<f32, String> {
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

fn load_dissect_input(pack_config: &PackConfig) -> Result<(DissectManifest, DissectInput), String> {
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

/// Verify the pack on disk, then write the emitted manifest. A failure here
/// leaves partial outputs that the caller's output guard rolls back.
fn verify_pack_and_emit_manifest(
    layout: &OutputLayout,
    bytes: &[u8],
    emitted: &Manifest,
) -> Result<io::PackStats, String> {
    let stats = io::verify_written_pack(&layout.pack_path, bytes)?;
    write_emitted_manifest(emitted, &layout.manifest_path)?;
    Ok(stats)
}

fn write_emitted_manifest(emitted: &Manifest, manifest_path: &Path) -> Result<(), String> {
    emitted
        .validate()
        .map_err(|e| format!("emitted manifest is invalid: {}", e))?;
    let serialized = serde_json::to_string_pretty(emitted)
        .map_err(|e| format!("Failed to serialize emitted manifest: {}", e))?;
    write_bytes_atomically(manifest_path, format!("{}\n", serialized).as_bytes()).map_err(|e| {
        format!(
            "Failed to write manifest '{}': {}",
            manifest_path.display(),
            e
        )
    })
}

fn register_and_persist(
    registry: &mut ArtifactRegistry,
    emitted: &Manifest,
    layout: &OutputLayout,
    recipe_id: &str,
    admission: PackAdmission,
    registry_file: &Path,
) -> Result<bool, String> {
    register_emitted(registry, emitted, layout, recipe_id, admission)
        .and_then(|()| persist_registry(registry, registry_file))
        .map(|()| admission == PackAdmission::ReplaceOwnPlanned)
}

fn register_emitted(
    registry: &mut ArtifactRegistry,
    emitted: &Manifest,
    layout: &OutputLayout,
    recipe_id: &str,
    admission: PackAdmission,
) -> Result<(), String> {
    let generated_status = emitted
        .generated_artifact
        .as_ref()
        .and_then(|artifact| artifact.status.as_deref());
    registry
        .commit_pack(
            emitted,
            ProducedRegistration {
                manifest_path: &layout.manifest_path,
                producer: PACK_GOZ1_PRODUCER,
                generated_status,
                recipe_id: Some(recipe_id),
            },
            admission,
        )
        .map_err(|e| {
            format!(
                "{}\n  note: registry registration failed, so this run was rolled back: \
                 outputs written by it were removed and any pre-existing outputs were \
                 restored. The registry key is the generated slug '{}' (always \
                 `<source model slug>_goz1`).",
                e, emitted.model.slug
            )
        })
}

/// Snapshots of output files that existed before this run. On failure,
/// [`OutputGuard::restore`] puts the previous bytes back so an existing
/// registry entry is never left pointing at deleted or half-written artifacts;
/// on success the temporary snapshots delete themselves.
struct OutputGuard {
    entries: Vec<GuardedOutput>,
}

struct GuardedOutput {
    dest: PathBuf,
    backup: Option<tempfile::NamedTempFile>,
}

impl OutputGuard {
    fn snapshot(layout: &OutputLayout) -> Result<Self, String> {
        let mut entries = Vec::with_capacity(2);
        for dest in [&layout.pack_path, &layout.manifest_path] {
            let backup = match std::fs::metadata(dest) {
                Ok(meta) if meta.is_file() => Some(snapshot_file(dest)?),
                _ => None,
            };
            entries.push(GuardedOutput {
                dest: dest.clone(),
                backup,
            });
        }
        Ok(Self { entries })
    }

    /// Roll back outputs; returns any restore/removal failures so the caller
    /// can surface them alongside the original error rather than hide them.
    fn restore(&self) -> Vec<String> {
        let mut failures = Vec::new();
        for entry in &self.entries {
            let result = match &entry.backup {
                Some(backup) => std::fs::copy(backup.path(), &entry.dest).map(|_| ()),
                None => match std::fs::remove_file(&entry.dest) {
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                    result => result,
                },
            };
            if let Err(e) = result {
                failures.push(format!("{}: {}", entry.dest.display(), e));
            }
        }
        failures
    }
}

fn snapshot_file(dest: &Path) -> Result<tempfile::NamedTempFile, String> {
    let parent = dest
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut backup = tempfile::NamedTempFile::new_in(parent).map_err(|e| {
        format!(
            "Failed to snapshot existing output '{}': {}",
            dest.display(),
            e
        )
    })?;
    let mut source = std::fs::File::open(dest).map_err(|e| {
        format!(
            "Failed to snapshot existing output '{}': {}",
            dest.display(),
            e
        )
    })?;
    std::io::copy(&mut source, backup.as_file_mut()).map_err(|e| {
        format!(
            "Failed to snapshot existing output '{}': {}",
            dest.display(),
            e
        )
    })?;
    Ok(backup)
}

fn persist_registry(registry: &ArtifactRegistry, registry_file: &Path) -> Result<(), String> {
    registry_io::store_registry(registry, registry_file)
}

#[cfg(test)]
mod tests;
