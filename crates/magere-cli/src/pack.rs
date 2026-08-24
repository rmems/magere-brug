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

use crate::checksum;
use crate::manifest::{
    BackendStatus, Checksum, GeneratedArtifact, Manifest, Metadata, Quantization, SourceLineage,
    TensorSummary,
};
use crate::registry::ArtifactRegistry;
use magere_grok_process::manifest::load_manifest;
use magere_grok_process::stream::run_quantize;
use magere_grok_process::types::{GOZ1_MAGIC, GOZ1_VERSION, InputFormat, QuantizeConfig};
use magere_grok_process::weight_pack::{TENSOR_F16, TENSOR_TERNARY, parse_pack};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Recipe `type` values this runner accepts.
const PACK_RECIPE_TYPES: &[&str] = &["goz1_pack", "ternary_pack"];

/// Verbatim notice recorded on every artifact this runner produces.
///
/// See the module docs: the packer does not load real weights yet, so nothing downstream may
/// treat these files as checkpoints.
pub const SKELETON_NOTICE: &str = "SKELETON PACK — magere-grok-process::stream::run_quantize does not load real tensor weights yet: every tensor payload is a 4-byte placeholder with shape [1, 1]. The file is a structurally valid GOZ1 shell, not a usable checkpoint, which is why generated_artifact.status is 'planned' rather than 'success'.";

/// `generated_artifact.status` recorded for skeleton packs.
///
/// `planned` is the only value in the manifest schema's status enum
/// (`planned|running|success|failed|skipped`) that does not assert a finished artifact while
/// still allowing `path`, `checksum` and `tensor_summary` to be recorded alongside it.
const SKELETON_STATUS: &str = "planned";

/// A recipe document as consumed by `magere pack-goz1`.
///
/// Unknown top-level keys are ignored on purpose so recipes carrying blocks owned by other
/// runners (e.g. a future `saaq` block) still load here; the `pack` block itself is strict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackRecipe {
    pub recipe_id: String,
    #[serde(rename = "type")]
    pub recipe_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs: Option<RecipeInputs>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outputs: Option<RecipeOutputs>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pack: Option<PackConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeInputs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_manifest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goz1_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeOutputs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_dir: Option<String>,
}

/// The `pack` block: everything needed to build a [`QuantizeConfig`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackConfig {
    /// Path to the xai-dissect `DissectManifest` JSON.
    pub dissect_manifest: String,
    /// Source weight directory. Defaults to the source manifest's `source_artifact.path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_dir: Option<String>,
    /// Defaults to a mapping of the source manifest's `source_artifact.format`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_format: Option<InputFormat>,
    /// GIF saliency threshold ratio; defaults to the `QuantizeConfig` default (0.05).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gif_threshold: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_embedded_baseline: Option<bool>,
}

/// Facts about the pack file on disk, derived by reading it back after the write.
#[derive(Debug, Clone, Copy)]
struct PackStats {
    tensor_count: u32,
    /// FP16-encoded tensors. Preserve-tier tensors share the FP16 on-disk encoding and are
    /// therefore counted here too.
    f16_count: u32,
    ternary_count: u32,
    size_bytes: u64,
}

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

/// CLI entry point: run the recipe and render a human-readable report.
pub fn pack_goz1_command(
    recipe_path: &Path,
    registry_path: Option<&Path>,
    output_dir_override: Option<&Path>,
) -> Result<String, String> {
    let outcome = run_pack_recipe(recipe_path, registry_path, output_dir_override)?;
    Ok(format_outcome(&outcome))
}

/// Parse a recipe file and reject anything this runner cannot execute.
pub fn load_pack_recipe(recipe_path: &Path) -> Result<PackRecipe, String> {
    let content = std::fs::read_to_string(recipe_path)
        .map_err(|e| format!("Failed to read recipe '{}': {}", recipe_path.display(), e))?;
    let recipe: PackRecipe = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse recipe '{}': {}", recipe_path.display(), e))?;

    if recipe.recipe_id.trim().is_empty() {
        return Err(format!(
            "recipe '{}' is missing recipe_id",
            recipe_path.display()
        ));
    }
    if !PACK_RECIPE_TYPES.contains(&recipe.recipe_type.as_str()) {
        return Err(format!(
            "recipe '{}' has type '{}'; `pack-goz1` only runs {}",
            recipe.recipe_id,
            recipe.recipe_type,
            PACK_RECIPE_TYPES.join(" or ")
        ));
    }
    if let Some(outputs) = &recipe.outputs
        && let Some(format) = &outputs.generated_format
        && format != "goz1"
    {
        return Err(format!(
            "recipe '{}' declares outputs.generated_format '{}'; `pack-goz1` only writes goz1",
            recipe.recipe_id, format
        ));
    }

    Ok(recipe)
}

/// Run a pack recipe end to end: quantize → write → verify → manifest → registry.
pub fn run_pack_recipe(
    recipe_path: &Path,
    registry_path: Option<&Path>,
    output_dir_override: Option<&Path>,
) -> Result<PackOutcome, String> {
    let recipe = load_pack_recipe(recipe_path)?;

    let pack_config = recipe.pack.as_ref().ok_or_else(|| {
        format!(
            "recipe '{}' has no `pack` block; `pack-goz1` needs one (see configs/recipes/ternary-pack-example.json)",
            recipe.recipe_id
        )
    })?;

    let source_manifest_path = recipe
        .inputs
        .as_ref()
        .and_then(|i| i.source_manifest.as_deref())
        .ok_or_else(|| {
            format!(
                "recipe '{}' is missing inputs.source_manifest",
                recipe.recipe_id
            )
        })?;
    let source = Manifest::from_file(source_manifest_path).map_err(|e| {
        format!(
            "Failed to load source manifest '{}': {}",
            source_manifest_path, e
        )
    })?;
    source.validate().map_err(|e| {
        format!(
            "source manifest '{}' is invalid: {}",
            source_manifest_path, e
        )
    })?;

    // --- Resolve output locations -------------------------------------------------------
    let output_dir: PathBuf = match output_dir_override {
        Some(dir) => dir.to_path_buf(),
        None => recipe
            .outputs
            .as_ref()
            .and_then(|o| o.output_dir.as_deref())
            .map(PathBuf::from)
            .ok_or_else(|| {
                format!(
                    "recipe '{}' has no outputs.output_dir; pass --output-dir",
                    recipe.recipe_id
                )
            })?,
    };

    let manifest_id = recipe
        .outputs
        .as_ref()
        .and_then(|o| o.manifest_id.clone())
        .unwrap_or_else(|| format!("{}-goz1", source.metadata.manifest_id));
    let stem = file_stem_for(&manifest_id)?;
    let pack_path = output_dir.join(format!("{}.goz1", stem));
    let manifest_path = output_dir.join(format!("{}.manifest.json", stem));

    // --- Build the QuantizeConfig -------------------------------------------------------
    let input_dir = pack_config
        .input_dir
        .clone()
        .unwrap_or_else(|| source.source_artifact.path.clone());
    let input_format = match pack_config.input_format {
        Some(format) => format,
        None => input_format_for(&source.source_artifact.format)?,
    };
    let gif_threshold = match pack_config.gif_threshold {
        Some(threshold) => {
            if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                return Err(format!(
                    "pack.gif_threshold must be a finite number in [0.0, 1.0] (got {})",
                    threshold
                ));
            }
            threshold
        }
        None => QuantizeConfig::default().gif_threshold,
    };
    let dissect_manifest_path = PathBuf::from(&pack_config.dissect_manifest);

    let quantize_config = QuantizeConfig {
        input_dir: input_dir.clone(),
        output_path: pack_path.display().to_string(),
        gif_threshold,
        input_format,
        manifest_path: Some(dissect_manifest_path.clone()),
        use_embedded_baseline: pack_config.use_embedded_baseline.unwrap_or(false),
    };

    // --- Pack ---------------------------------------------------------------------------
    let dissect = load_manifest(&dissect_manifest_path).map_err(|e| {
        format!(
            "Failed to load dissect manifest '{}': {}",
            dissect_manifest_path.display(),
            e
        )
    })?;
    if dissect.ternary_candidates.is_empty() {
        return Err(format!(
            "dissect manifest '{}' lists no ternary_candidates; refusing to write an empty GOZ1 pack",
            dissect_manifest_path.display()
        ));
    }

    let bytes =
        run_quantize(&quantize_config, &dissect).map_err(|e| format!("GOZ1 pack failed: {}", e))?;

    std::fs::create_dir_all(&output_dir).map_err(|e| {
        format!(
            "Failed to create output dir '{}': {}",
            output_dir.display(),
            e
        )
    })?;
    std::fs::write(&pack_path, &bytes)
        .map_err(|e| format!("Failed to write pack '{}': {}", pack_path.display(), e))?;

    // --- Verify the file we just wrote round-trips as GOZ1 ------------------------------
    let stats = verify_written_pack(&pack_path)?;

    let sha256 = checksum::compute_file_sha256(&pack_path)
        .map_err(|e| format!("Failed to checksum '{}': {}", pack_path.display(), e))?;

    // --- Emit the manifest --------------------------------------------------------------
    let emitted =
        build_generated_manifest(&recipe, &source, manifest_id, &pack_path, &sha256, stats);
    emitted
        .validate()
        .map_err(|e| format!("emitted manifest is invalid: {}", e))?;

    let serialized = serde_json::to_string_pretty(&emitted)
        .map_err(|e| format!("Failed to serialize emitted manifest: {}", e))?;
    std::fs::write(&manifest_path, format!("{}\n", serialized)).map_err(|e| {
        format!(
            "Failed to write manifest '{}': {}",
            manifest_path.display(),
            e
        )
    })?;

    // --- Register -----------------------------------------------------------------------
    let registry_file = registry_path.unwrap_or_else(|| Path::new("registry.json"));
    let mut registry = if registry_file.exists() {
        let content = std::fs::read_to_string(registry_file)
            .map_err(|e| format!("Failed to read registry: {}", e))?;
        ArtifactRegistry::from_json(&content)
            .map_err(|e| format!("Failed to parse registry: {}", e))?
    } else {
        ArtifactRegistry::new()
    };
    // Re-running the same recipe replaces its own entry rather than tripping the registry's
    // unique-slug rule.
    let registry_replaced = registry.models.remove(&emitted.model.slug).is_some();
    registry.register(&emitted)?;
    let registry_json = registry
        .to_json_pretty()
        .map_err(|e| format!("Failed to serialize registry: {}", e))?;
    std::fs::write(registry_file, registry_json)
        .map_err(|e| format!("Failed to write registry: {}", e))?;

    Ok(PackOutcome {
        recipe_id: recipe.recipe_id,
        pack_path,
        manifest_path,
        registry_path: registry_file.to_path_buf(),
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

/// Read the pack back from disk and assert it is a well-formed GOZ1 file.
fn verify_written_pack(pack_path: &Path) -> Result<PackStats, String> {
    let written = std::fs::read(pack_path)
        .map_err(|e| format!("Failed to re-read pack '{}': {}", pack_path.display(), e))?;

    if !written.starts_with(GOZ1_MAGIC) {
        return Err(format!(
            "pack '{}' does not start with the GOZ1 magic",
            pack_path.display()
        ));
    }
    let (header, entries) = parse_pack(&written).ok_or_else(|| {
        format!(
            "pack '{}' did not round-trip through parse_pack",
            pack_path.display()
        )
    })?;
    if header.magic != *GOZ1_MAGIC {
        return Err(format!(
            "pack '{}' has magic {:?}, expected GOZ1",
            pack_path.display(),
            header.magic
        ));
    }
    if header.version != GOZ1_VERSION {
        return Err(format!(
            "pack '{}' has version {}, expected {}",
            pack_path.display(),
            header.version,
            GOZ1_VERSION
        ));
    }
    if header.tensor_count as usize != entries.len() {
        return Err(format!(
            "pack '{}' header claims {} tensors but the table holds {}",
            pack_path.display(),
            header.tensor_count,
            entries.len()
        ));
    }

    let mut f16_count: u32 = 0;
    let mut ternary_count: u32 = 0;
    for entry in &entries {
        match entry.dtype {
            TENSOR_F16 => f16_count += 1,
            TENSOR_TERNARY => ternary_count += 1,
            other => {
                return Err(format!(
                    "pack '{}' tensor '{}' has unknown dtype 0x{:02x}",
                    pack_path.display(),
                    entry.name,
                    other
                ));
            }
        }
    }

    Ok(PackStats {
        tensor_count: header.tensor_count,
        f16_count,
        ternary_count,
        size_bytes: written.len() as u64,
    })
}

/// Build the model manifest that records the pack as a `generated_artifact`.
fn build_generated_manifest(
    recipe: &PackRecipe,
    source: &Manifest,
    manifest_id: String,
    pack_path: &Path,
    sha256: &str,
    stats: PackStats,
) -> Manifest {
    let now = chrono::Utc::now().to_rfc3339();

    let mut model = source.model.clone();
    // Keep the pack distinct from its source in the registry's slug-keyed map.
    model.slug = format!("{}_goz1", source.model.slug);

    let recipe_note = match &recipe.description {
        Some(description) => format!(" Recipe description: {}", description),
        None => String::new(),
    };

    let mut backends = HashMap::new();
    backends.insert(
        "goz1".to_string(),
        BackendStatus {
            supported: Some(true),
            status: Some("planned".to_string()),
            kernel_types: None,
        },
    );

    Manifest {
        metadata: Metadata {
            schema_version: source.metadata.schema_version,
            created_at: now.clone(),
            manifest_id,
            description: Some(format!(
                "GOZ1 pack emitted by `magere pack-goz1` from recipe '{}'.{} {}",
                recipe.recipe_id, recipe_note, SKELETON_NOTICE
            )),
        },
        model,
        source_artifact: source.source_artifact.clone(),
        generated_artifact: Some(GeneratedArtifact {
            format: "goz1".to_string(),
            path: Some(pack_path.display().to_string()),
            status: Some(SKELETON_STATUS.to_string()),
            version: Some(GOZ1_VERSION),
            source_url: None,
            checksum: Some(Checksum {
                sha256: Some(sha256.to_string()),
                md5: None,
            }),
            dtype_summary: None,
            size_bytes: Some(stats.size_bytes),
            shard_info: None,
            timestamp: Some(now),
            source_lineage: Some(SourceLineage {
                manifest_id: Some(source.metadata.manifest_id.clone()),
                path: Some(source.source_artifact.path.clone()),
                checksum: source.source_artifact.checksum.clone(),
            }),
            tensor_summary: Some(TensorSummary {
                tensor_count: Some(stats.tensor_count),
                f16_count: Some(stats.f16_count),
                ternary_count: Some(stats.ternary_count),
            }),
        }),
        quantization: Some(Quantization {
            method: Some("ternary".to_string()),
            bits: Some(2),
            group_size: None,
            calibration_dataset: source
                .quantization
                .as_ref()
                .and_then(|q| q.calibration_dataset.clone()),
            calibration_config_path: source
                .quantization
                .as_ref()
                .and_then(|q| q.calibration_config_path.clone()),
        }),
        backend_compatibility: Some(backends),
        saaq_experiment: None,
        benchmark_linkage: None,
    }
}

/// Map a manifest `source_artifact.format` onto a packer input format.
///
/// GGUF and `hf_repo` are registry source formats but not packer inputs; such recipes must
/// name `pack.input_format` explicitly.
fn input_format_for(source_format: &str) -> Result<InputFormat, String> {
    match source_format {
        "safetensors" => Ok(InputFormat::Safetensors),
        // NPY directories are recorded as local_dir in manifests (npy_dir is not a valid
        // source_artifact.format).
        "local_dir" => Ok(InputFormat::NpyDir),
        other => Err(format!(
            "source_artifact.format '{}' is not a packer input; set pack.input_format to safetensors or npy_dir",
            other
        )),
    }
}

/// Derive a filesystem-safe file stem from a manifest id.
fn file_stem_for(manifest_id: &str) -> Result<&str, String> {
    let trimmed = manifest_id.trim();
    if trimmed.is_empty() {
        return Err("outputs.manifest_id must not be empty".to_string());
    }
    if trimmed.contains('/') || trimmed.contains('\\') || trimmed.contains("..") {
        return Err(format!(
            "outputs.manifest_id '{}' must not contain path separators or '..'",
            manifest_id
        ));
    }
    Ok(trimmed)
}

/// Render the human-readable CLI report, skeleton caveat included.
fn format_outcome(outcome: &PackOutcome) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "✓ GOZ1 pack written from recipe '{}': {}\n",
        outcome.recipe_id,
        outcome.pack_path.display()
    ));
    out.push_str(&format!(
        "  magic: GOZ1  version: {}  tensors: {} (ternary {}, f16/preserve {})  size: {} bytes\n",
        GOZ1_VERSION,
        outcome.tensor_count,
        outcome.ternary_count,
        outcome.f16_count,
        outcome.size_bytes
    ));
    out.push_str(&format!("  sha256: {}\n", outcome.sha256));
    out.push_str(&format!(
        "  manifest: {} (manifest_id: {}, slug: {}, generated_artifact.status: {})\n",
        outcome.manifest_path.display(),
        outcome.manifest.metadata.manifest_id,
        outcome.manifest.model.slug,
        outcome
            .manifest
            .generated_artifact
            .as_ref()
            .and_then(|g| g.status.as_deref())
            .unwrap_or("-")
    ));
    out.push_str(&format!(
        "  registry: {} ({})\n",
        outcome.registry_path.display(),
        if outcome.registry_replaced {
            "entry replaced"
        } else {
            "entry added"
        }
    ));
    if outcome.input_dir_missing {
        out.push_str(
            "  note: pack.input_dir does not exist on disk; the skeleton packer never reads it\n",
        );
    }
    out.push_str(&format!("! {}\n", SKELETON_NOTICE));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;

    const FIXTURE_DISSECT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/recipes/fixtures/grok-mini-dissect.json"
    );
    const SOURCE_MANIFEST: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../manifests/examples/redpajama-incite-7b-chat.json"
    );
    const EXAMPLE_RECIPE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/recipes/ternary-pack-example.json"
    );
    const REPO_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

    /// The fixture lists 6 ternary_candidates: 3 stay ternary, 2 match the `norm` fp16 rule
    /// and 1 matches the `router` preserve rule (both encode as TENSOR_F16 on disk).
    const FIXTURE_TENSORS: u32 = 6;
    const FIXTURE_TERNARY: u32 = 3;
    const FIXTURE_F16: u32 = 3;

    struct Harness {
        dir: TempDir,
        recipe_path: PathBuf,
        output_dir: PathBuf,
        registry_path: PathBuf,
    }

    fn write_recipe(dir: &Path, name: &str, recipe: &serde_json::Value) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, serde_json::to_string_pretty(recipe).unwrap()).unwrap();
        path
    }

    /// Wrap an already-created temp dir plus a recipe whose `outputs.output_dir` is
    /// `<dir>/packs`, so `Harness::output_dir` really is where the run will write.
    fn harness_from(dir: TempDir, recipe: &serde_json::Value) -> Harness {
        let output_dir = dir.path().join("packs");
        let registry_path = dir.path().join("registry.json");
        let recipe_path = write_recipe(dir.path(), "recipe.json", recipe);
        Harness {
            dir,
            recipe_path,
            output_dir,
            registry_path,
        }
    }

    fn base_recipe(output_dir: &Path) -> serde_json::Value {
        json!({
            "recipe_id": "test-ternary-pack",
            "type": "ternary_pack",
            "description": "unit-test recipe",
            "inputs": { "source_manifest": SOURCE_MANIFEST },
            "outputs": {
                "generated_format": "goz1",
                "manifest_id": "test-pack-v1",
                "output_dir": output_dir.display().to_string()
            },
            "pack": {
                "dissect_manifest": FIXTURE_DISSECT,
                "input_dir": "/models/redpajama/INCITE-7B-Chat",
                "input_format": "safetensors",
                "gif_threshold": 0.05,
                "use_embedded_baseline": false
            }
        })
    }

    fn default_harness() -> Harness {
        let dir = TempDir::new().unwrap();
        let recipe = base_recipe(&dir.path().join("packs"));
        harness_from(dir, &recipe)
    }

    #[test]
    fn end_to_end_writes_parseable_goz1_and_registers_it() {
        let h = default_harness();
        let outcome =
            run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).expect("pack run");

        // The pack file exists, starts with the magic, and parses as GOZ1.
        let bytes = fs::read(&outcome.pack_path).unwrap();
        assert!(bytes.starts_with(b"GOZ1"), "pack must start with GOZ1");
        let (header, entries) = parse_pack(&bytes).expect("parse_pack");
        assert_eq!(header.version, GOZ1_VERSION);
        assert_eq!(header.tensor_count, FIXTURE_TENSORS);
        assert_eq!(entries.len() as u32, FIXTURE_TENSORS);
        assert_eq!(outcome.tensor_count, FIXTURE_TENSORS);
        assert_eq!(outcome.ternary_count, FIXTURE_TERNARY);
        assert_eq!(outcome.f16_count, FIXTURE_F16);
        assert_eq!(outcome.size_bytes, bytes.len() as u64);
        assert_eq!(
            outcome.pack_path,
            h.output_dir.join("test-pack-v1.goz1"),
            "pack file name derives from outputs.manifest_id"
        );

        // Checksum matches what we recorded.
        assert!(
            checksum::verify_checksum(&outcome.pack_path, &outcome.sha256).unwrap(),
            "recorded sha256 must match the written file"
        );

        // The emitted manifest validates and is honest about the skeleton.
        assert!(outcome.manifest.validate().is_ok());
        let generated = outcome.manifest.generated_artifact.as_ref().unwrap();
        assert_eq!(generated.format, "goz1");
        assert_eq!(generated.status.as_deref(), Some("planned"));
        assert_eq!(generated.version, Some(1));
        assert_eq!(
            generated.checksum.as_ref().unwrap().sha256.as_deref(),
            Some(outcome.sha256.as_str())
        );
        let summary = generated.tensor_summary.as_ref().unwrap();
        assert_eq!(summary.tensor_count, Some(FIXTURE_TENSORS));
        assert_eq!(summary.f16_count, Some(FIXTURE_F16));
        assert_eq!(summary.ternary_count, Some(FIXTURE_TERNARY));
        let lineage = generated.source_lineage.as_ref().unwrap();
        assert_eq!(
            lineage.manifest_id.as_deref(),
            Some("redpajama-incite-7b-chat-v1")
        );
        assert!(
            outcome
                .manifest
                .metadata
                .description
                .as_deref()
                .unwrap()
                .contains("SKELETON PACK")
        );

        // The manifest was written to disk and re-validates from there.
        let reloaded = Manifest::from_file(&outcome.manifest_path).expect("reload manifest");
        assert!(reloaded.validate().is_ok());
        assert_eq!(reloaded.metadata.manifest_id, "test-pack-v1");
        assert_eq!(reloaded.model.slug, "redpajama_incite_7b_chat_goz1");

        // The registry gained the entry.
        let registry =
            ArtifactRegistry::from_json(&fs::read_to_string(&outcome.registry_path).unwrap())
                .unwrap();
        let entry = registry
            .lookup("redpajama_incite_7b_chat_goz1")
            .expect("registry entry");
        assert_eq!(entry.manifest_id, "test-pack-v1");
        assert!(!outcome.registry_replaced);

        // The rendered CLI report carries the caveat.
        let report = format_outcome(&outcome);
        assert!(report.contains("SKELETON PACK"));
        assert!(report.contains("placeholder"));
    }

    #[test]
    fn rerunning_the_same_recipe_replaces_the_registry_entry() {
        let h = default_harness();
        run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap();
        let second = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap();
        assert!(second.registry_replaced);
        let registry =
            ArtifactRegistry::from_json(&fs::read_to_string(&h.registry_path).unwrap()).unwrap();
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn output_dir_override_wins_over_the_recipe() {
        let h = default_harness();
        let elsewhere = h.dir.path().join("elsewhere");
        let outcome = run_pack_recipe(
            &h.recipe_path,
            Some(&h.registry_path),
            Some(elsewhere.as_path()),
        )
        .unwrap();
        assert_eq!(outcome.pack_path, elsewhere.join("test-pack-v1.goz1"));
        assert!(outcome.pack_path.exists());
        assert!(outcome.manifest_path.exists());
        assert!(!h.output_dir.exists(), "recipe output_dir stays untouched");
    }

    #[test]
    fn tracks_whether_input_dir_exists() {
        let dir = TempDir::new().unwrap();
        let present = dir.path().join("weights");
        fs::create_dir_all(&present).unwrap();
        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["pack"]["input_dir"] = json!(present.display().to_string());
        let h = harness_from(dir, &recipe);
        let outcome = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap();
        assert!(!outcome.input_dir_missing);
        assert!(!format_outcome(&outcome).contains("does not exist on disk"));

        let dir = TempDir::new().unwrap();
        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["pack"]["input_dir"] = json!(dir.path().join("gone").display().to_string());
        let h = harness_from(dir, &recipe);
        let outcome = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap();
        assert!(outcome.input_dir_missing);
        assert!(format_outcome(&outcome).contains("does not exist on disk"));
    }

    /// The shipped example uses repo-root-relative paths (recipe paths resolve against the
    /// process working directory). Unit tests run with the crate directory as cwd, so the
    /// references are resolved against the repo root here and the recipe is re-materialised
    /// with absolute paths before being run.
    #[test]
    fn shipped_example_recipe_is_well_formed_and_runnable() {
        let recipe = load_pack_recipe(Path::new(EXAMPLE_RECIPE)).expect("example recipe parses");
        assert_eq!(recipe.recipe_type, "ternary_pack");
        let pack = recipe.pack.as_ref().expect("example carries a pack block");
        assert_eq!(pack.input_format, Some(InputFormat::Safetensors));

        let root = Path::new(REPO_ROOT);
        let source_manifest = root.join(
            recipe
                .inputs
                .as_ref()
                .unwrap()
                .source_manifest
                .as_ref()
                .unwrap(),
        );
        let dissect = root.join(&pack.dissect_manifest);
        assert!(source_manifest.exists(), "{}", source_manifest.display());
        assert!(dissect.exists(), "{}", dissect.display());

        let dir = TempDir::new().unwrap();
        let mut as_absolute: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(EXAMPLE_RECIPE).unwrap()).unwrap();
        as_absolute["inputs"]["source_manifest"] = json!(source_manifest.display().to_string());
        as_absolute["pack"]["dissect_manifest"] = json!(dissect.display().to_string());
        let recipe_path = write_recipe(dir.path(), "example.json", &as_absolute);
        let registry_path = dir.path().join("registry.json");

        let outcome = run_pack_recipe(&recipe_path, Some(&registry_path), Some(dir.path()))
            .expect("shipped example recipe must run");
        assert!(fs::read(&outcome.pack_path).unwrap().starts_with(b"GOZ1"));
        assert_eq!(outcome.tensor_count, FIXTURE_TENSORS);
        assert_eq!(
            outcome.pack_path,
            dir.path().join("redpajama-incite-7b-chat-goz1-v1.goz1")
        );
        assert_eq!(
            outcome
                .manifest
                .generated_artifact
                .as_ref()
                .unwrap()
                .status
                .as_deref(),
            Some("planned")
        );
    }

    #[test]
    fn goz1_pack_type_is_accepted_too() {
        let dir = TempDir::new().unwrap();
        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["type"] = json!("goz1_pack");
        let h = harness_from(dir, &recipe);
        let outcome = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap();
        assert_eq!(outcome.tensor_count, FIXTURE_TENSORS);
    }

    #[test]
    fn rejects_non_pack_recipe_type() {
        let dir = TempDir::new().unwrap();
        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["type"] = json!("saaq");
        let h = harness_from(dir, &recipe);
        let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
        assert!(err.contains("saaq"), "{}", err);
        assert!(err.contains("goz1_pack"), "{}", err);
    }

    #[test]
    fn rejects_missing_pack_block() {
        let dir = TempDir::new().unwrap();
        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe.as_object_mut().unwrap().remove("pack");
        let h = harness_from(dir, &recipe);
        let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
        assert!(err.contains("`pack` block"), "{}", err);
    }

    #[test]
    fn rejects_unknown_pack_key() {
        let dir = TempDir::new().unwrap();
        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["pack"]["bit_width"] = json!(2);
        let h = harness_from(dir, &recipe);
        let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
        assert!(err.contains("bit_width"), "{}", err);
    }

    #[test]
    fn rejects_missing_dissect_manifest() {
        let dir = TempDir::new().unwrap();
        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["pack"]["dissect_manifest"] = json!("/nonexistent/dissect.json");
        let h = harness_from(dir, &recipe);
        let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
        assert!(err.contains("dissect manifest"), "{}", err);
        assert!(!h.output_dir.exists(), "nothing is written on failure");
        assert!(
            !h.registry_path.exists(),
            "registry is untouched on failure"
        );
    }

    #[test]
    fn rejects_missing_source_manifest() {
        let dir = TempDir::new().unwrap();
        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["inputs"] = json!({});
        let h = harness_from(dir, &recipe);
        let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
        assert!(err.contains("inputs.source_manifest"), "{}", err);
    }

    #[test]
    fn rejects_unwritable_output_dir() {
        let dir = TempDir::new().unwrap();
        // A regular file where the output directory should be: create_dir_all must fail.
        let blocker = dir.path().join("blocked");
        fs::write(&blocker, b"not a directory").unwrap();
        let recipe = base_recipe(&blocker.join("packs"));
        let h = harness_from(dir, &recipe);
        let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
        assert!(err.contains("Failed to create output dir"), "{}", err);
    }

    #[test]
    fn rejects_missing_output_dir_without_override() {
        let dir = TempDir::new().unwrap();
        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["outputs"]
            .as_object_mut()
            .unwrap()
            .remove("output_dir");
        let h = harness_from(dir, &recipe);
        let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
        assert!(err.contains("--output-dir"), "{}", err);
    }

    #[test]
    fn rejects_out_of_range_gif_threshold() {
        let dir = TempDir::new().unwrap();
        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["pack"]["gif_threshold"] = json!(4.2);
        let h = harness_from(dir, &recipe);
        let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
        assert!(err.contains("gif_threshold"), "{}", err);
    }

    #[test]
    fn rejects_non_goz1_generated_format() {
        let dir = TempDir::new().unwrap();
        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["outputs"]["generated_format"] = json!("gguf");
        let h = harness_from(dir, &recipe);
        let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
        assert!(err.contains("generated_format"), "{}", err);
    }

    #[test]
    fn rejects_manifest_id_with_path_separators() {
        let dir = TempDir::new().unwrap();
        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["outputs"]["manifest_id"] = json!("../escape");
        let h = harness_from(dir, &recipe);
        let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
        assert!(err.contains("path separators"), "{}", err);
    }

    #[test]
    fn rejects_dissect_manifest_without_ternary_candidates() {
        let dir = TempDir::new().unwrap();
        let mut dissect: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(FIXTURE_DISSECT).unwrap()).unwrap();
        dissect["ternary_candidates"] = json!([]);
        let dissect_path = dir.path().join("empty-dissect.json");
        fs::write(&dissect_path, dissect.to_string()).unwrap();

        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["pack"]["dissect_manifest"] = json!(dissect_path.display().to_string());
        let h = harness_from(dir, &recipe);
        let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
        assert!(err.contains("ternary_candidates"), "{}", err);
    }

    #[test]
    fn rejects_malformed_recipe_json() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("broken.json");
        fs::write(&path, b"{ not json").unwrap();
        let err = load_pack_recipe(&path).unwrap_err();
        assert!(err.contains("Failed to parse recipe"), "{}", err);
    }

    #[test]
    fn input_format_defaults_follow_the_source_artifact() {
        assert_eq!(
            input_format_for("safetensors").unwrap(),
            InputFormat::Safetensors
        );
        assert_eq!(input_format_for("local_dir").unwrap(), InputFormat::NpyDir);
        let err = input_format_for("gguf").unwrap_err();
        assert!(err.contains("pack.input_format"), "{}", err);
    }

    #[test]
    fn gguf_source_needs_an_explicit_input_format() {
        let dir = TempDir::new().unwrap();
        let mut source: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(SOURCE_MANIFEST).unwrap()).unwrap();
        source["source_artifact"] = json!({
            "format": "gguf",
            "path": "/models/redpajama/INCITE-7B-Chat-F16.gguf"
        });
        let source_path = dir.path().join("gguf-source.json");
        fs::write(&source_path, source.to_string()).unwrap();

        let mut recipe = base_recipe(&dir.path().join("packs"));
        recipe["inputs"]["source_manifest"] = json!(source_path.display().to_string());
        recipe["pack"]
            .as_object_mut()
            .unwrap()
            .remove("input_format");
        let recipe_path = write_recipe(dir.path(), "gguf-default.json", &recipe);
        let registry_path = dir.path().join("registry.json");
        let err = run_pack_recipe(&recipe_path, Some(&registry_path), None).unwrap_err();
        assert!(err.contains("not a packer input"), "{}", err);

        // …and it succeeds once the recipe names a packer input explicitly.
        recipe["pack"]["input_format"] = json!("npy_dir");
        let recipe_path = write_recipe(dir.path(), "gguf-explicit.json", &recipe);
        let outcome = run_pack_recipe(&recipe_path, Some(&registry_path), None).unwrap();
        assert_eq!(outcome.tensor_count, FIXTURE_TENSORS);
    }

    #[test]
    fn pack_goz1_command_renders_a_report() {
        let h = default_harness();
        let report = pack_goz1_command(&h.recipe_path, Some(&h.registry_path), None).unwrap();
        assert!(report.contains("GOZ1 pack written"));
        assert!(report.contains("generated_artifact.status: planned"));
        assert!(report.contains("SKELETON PACK"));
    }
}
