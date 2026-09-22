//! Recipe pipeline foundation.
//!
//! A recipe is a small JSON document under `configs/recipes/` that names what to
//! register, pack, optionally export to GGUF, or calibrate. It never carries
//! weights; it points at manifests and records the lineage of whatever the run
//! produces.
//!
//! This module owns the recipe structure, the loader, the validator, the
//! `register` runner (which emits artifact manifests and combine-for-AI handoff
//! files), and a thin apply dispatch for `saaq` recipes that calls
//! [`crate::saaq::run_saaq_command`]. Execution of the remaining types is
//! deliberately not implemented here:
//!
//! - `goz1_pack` / `ternary_pack` -> issue #19 (`magere pack-goz1`)
//! - `gguf_export` -> optional conversion placeholder; existing GGUF artifacts
//!   are registered without conversion
//!
//! AWQ and GPTQ are removed comparison paths and are rejected by the schema.

mod apply;
mod resolve;
mod summary;
mod validate;

use clap::Subcommand;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Source of truth for the recipe shape, embedded so the CLI never depends on
/// the working directory to find `schemas/recipe.schema.json`.
const RECIPE_SCHEMA: &str = include_str!("../../../../schemas/recipe.schema.json");

/// GOZ1 pack format version written by `magere-grok-process`.
const SUPPORTED_GOZ1_VERSION: u32 = 1;

/// Registry used when neither `--registry` nor `outputs.registry_path` is set.
const DEFAULT_REGISTRY_PATH: &str = "registry.json";

const VALID_GENERATED_FORMATS: &[&str] = &["goz1", "gguf", "ternary", "binary"];
const VALID_SOURCE_FORMATS: &[&str] = &["gguf", "safetensors", "hf_repo", "local_dir"];
/// Source formats `magere-grok-process` can actually pack. NPY directories are
/// recorded as `local_dir` and mapped to `InputFormat::NpyDir`; GGUF stays a
/// registry/routing source format and is not a packer input. See
/// `docs/ARCHITECTURE.md` ("Primary path").
const PACKABLE_SOURCE_FORMATS: &[&str] = &["safetensors", "local_dir"];
/// Source formats that an optional GGUF export recipe may convert from.
/// Existing GGUF files skip this step and use `type: register`.
const EXPORTABLE_SOURCE_FORMATS: &[&str] = &["safetensors", "hf_repo", "local_dir"];

/// `magere recipe <...>` subcommands.
///
/// Kept in this module so `main.rs` only needs a single additive enum variant.
#[derive(Subcommand)]
pub enum RecipeCommands {
    /// Validate a recipe against the recipe JSON schema and its manifest references
    Validate {
        /// Path to recipe JSON file
        #[arg(value_name = "FILE")]
        path: PathBuf,
    },
    /// Inspect a recipe
    Inspect {
        /// Path to recipe JSON file
        #[arg(value_name = "FILE")]
        path: PathBuf,
    },
    /// Apply a recipe. `register` writes manifests/registry/handoff; `saaq` delegates to `run-saaq`
    Apply {
        /// Path to recipe JSON file
        #[arg(value_name = "FILE")]
        path: PathBuf,

        /// Path to save registry (overrides `outputs.registry_path`)
        #[arg(short, long)]
        registry: Option<PathBuf>,
    },
}

/// Recipe kind. Mirrors the `type` enum in `schemas/recipe.schema.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeType {
    /// Register an existing source artifact (GGUF, safetensors, HF repo, local dir).
    Register,
    /// Pack a source into a GOZ1 artifact. Runner: issue #19.
    Goz1Pack,
    /// Ternary weight pack (normally emitted as GOZ1). Runner: issue #19.
    TernaryPack,
    /// Optional GGUF conversion from a safetensors/HF/local-dir source. Placeholder.
    GgufExport,
    /// SAAQ validation run over a source or a registered GOZ1 pack.
    Saaq,
}

impl RecipeType {
    fn as_str(&self) -> &'static str {
        match self {
            RecipeType::Register => "register",
            RecipeType::Goz1Pack => "goz1_pack",
            RecipeType::TernaryPack => "ternary_pack",
            RecipeType::GgufExport => "gguf_export",
            RecipeType::Saaq => "saaq",
        }
    }

    fn is_pack(&self) -> bool {
        matches!(self, RecipeType::Goz1Pack | RecipeType::TernaryPack)
    }
}

impl std::fmt::Display for RecipeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A pipeline recipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub recipe_id: String,
    #[serde(rename = "type")]
    pub recipe_type: RecipeType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inputs: Option<RecipeInputs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<RecipeOutputs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handoff: Option<Handoff>,
    /// SAAQ runner configuration. Validated by the schema; executed by
    /// [`crate::saaq::run_saaq_command`]. Stored as JSON so this module does
    /// not re-parse the SAAQ-specific knobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saaq: Option<Value>,
    /// Path the recipe was loaded from. Never serialized; used to resolve
    /// relative manifest references without depending on the caller's cwd.
    #[serde(skip)]
    source_path: Option<PathBuf>,
    /// Original JSON document, retained so schema validation sees explicit
    /// `null` values that typed deserialization would otherwise drop.
    #[serde(skip)]
    source_json: Option<Value>,
    /// Exact UTF-8 text passed to [`Self::from_json`]. SAAQ apply hashes this
    /// buffer so `run_manifest.json` matches `magere run-saaq` and the file.
    #[serde(skip)]
    source_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeInputs {
    /// Manifest path (`*.json`, resolved and parsed). Registry ids are reserved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_manifest: Option<String>,
    /// Asserted `source_artifact.format` of the referenced manifest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_format: Option<String>,
    /// Manifest carrying a registered GOZ1 pack. Registry ids are reserved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub goz1_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeOutputs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub goz1_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum_algorithm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub register: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registry_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lineage: Option<RecipeLineage>,
}

/// Provenance recorded on an emitted artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeLineage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_manifest_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipe_id: Option<String>,
}

/// Calibration config for pack and export recipes that consume calibration data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationConfig {
    pub dataset: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Forward-declared handoff placeholders. magere-brug never executes these.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Handoff {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub myelin_accelerator: Option<HandoffTarget>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corinth_canal: Option<HandoffTarget>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub combine_for_ai: Option<HandoffTarget>,
}

impl Handoff {
    /// Declared targets, in a stable order, paired with the owning repo name.
    fn targets(&self) -> Vec<(&'static str, &HandoffTarget)> {
        let mut targets = Vec::new();
        if let Some(target) = &self.myelin_accelerator {
            targets.push(("myelin_accelerator", target));
        }
        if let Some(target) = &self.corinth_canal {
            targets.push(("corinth_canal", target));
        }
        if let Some(target) = &self.combine_for_ai {
            targets.push(("combine_for_ai", target));
        }
        targets
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffTarget {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_types: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl HandoffTarget {
    pub(super) fn effective_enabled(&self) -> bool {
        self.enabled.unwrap_or(false)
    }

    pub(super) fn effective_status(&self) -> &str {
        match (self.effective_enabled(), self.status.as_deref()) {
            (false, Some("ready") | None) => "placeholder",
            (true, None) => "ready",
            (_, Some(status)) => status,
        }
    }
}

impl Recipe {
    /// Load a recipe from a JSON string. Relative references resolve against cwd.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let value: Value = serde_json::from_str(json)?;
        let mut recipe: Self = serde_json::from_value(value.clone())?;
        recipe.source_json = Some(value);
        recipe.source_text = Some(json.to_string());
        Ok(recipe)
    }

    /// Load a recipe from a file. Relative references resolve against the recipe
    /// file's directory (and its ancestors) before falling back to cwd.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        let mut recipe = Self::from_json(&content)?;
        recipe.source_path = Some(path.to_path_buf());
        Ok(recipe)
    }

    /// Validate the recipe against `schemas/recipe.schema.json`, then apply the
    /// semantic checks the schema cannot express (manifest references resolve
    /// and parse, cross-field consistency). AWQ/GPTQ rejection is enforced by the
    /// schema itself: every string leaf is a `safe_string` (which pattern-bans them)
    /// or a closed enum, and unknown keys are refused by `additionalProperties: false`.
    pub fn validate(&self) -> Result<(), String> {
        let owned;
        let instance = if let Some(value) = &self.source_json {
            value
        } else {
            owned = serde_json::to_value(self)
                .map_err(|e| format!("failed to serialize recipe for validation: {e}"))?;
            &owned
        };

        crate::saaq::reject_explicit_nulls(instance, "")?;
        resolve::validate_against_schema(instance)?;
        self.validate_semantics()
    }

    /// Execute the recipe.
    ///
    /// `register` writes the source manifest into the artifact registry, copies
    /// the manifest, and emits a combine-for-AI handoff file. `saaq` delegates
    /// to [`crate::saaq::run_saaq_command`]. The remaining types return an error
    /// naming the issue that owns their runner (or stating they are placeholders).
    pub fn apply(&self, registry_override: Option<&Path>) -> Result<String, String> {
        self.validate()?;

        match self.recipe_type {
            RecipeType::Register => self.apply_register(registry_override),
            RecipeType::Goz1Pack => Err(format!(
                "recipe '{}': goz1_pack execution is not implemented in this command \
                 — see issue #19 (magere pack-goz1)",
                self.recipe_id
            )),
            RecipeType::TernaryPack => Err(format!(
                "recipe '{}': ternary_pack execution is not implemented in this command \
                 — see issue #19 (magere pack-goz1)",
                self.recipe_id
            )),
            RecipeType::GgufExport => Err(format!(
                "recipe '{}': gguf_export is an optional conversion placeholder — \
                 existing GGUF artifacts should be registered without conversion \
                 (type: register). Conversion from safetensors/HF is not implemented here",
                self.recipe_id
            )),
            RecipeType::Saaq => self.apply_saaq(registry_override),
        }
    }

    fn apply_saaq(&self, registry_override: Option<&Path>) -> Result<String, String> {
        if registry_override.is_some() {
            return Err(format!(
                "recipe '{}': saaq apply does not write a registry; omit --registry",
                self.recipe_id
            ));
        }
        let path = self.source_path.as_ref().ok_or_else(|| {
            format!(
                "recipe '{}': saaq apply requires a recipe file path",
                self.recipe_id
            )
        })?;
        let contents = self.loaded_recipe_text(path)?;
        crate::saaq::run_saaq_from_json(&contents, path, None)
    }

    /// Prefer the bytes that produced this `Recipe`; never re-serialize JSON.
    fn loaded_recipe_text(&self, path: &Path) -> Result<String, String> {
        if let Some(text) = &self.source_text {
            return Ok(text.clone());
        }
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read recipe: {e}"))
    }

    /// Effective value of `outputs.register`. A register recipe defaults to `true`
    /// because registering its source manifest is the whole point of the type;
    /// every other type defaults to `false`. `inspect` and `apply` must not disagree.
    fn registers_output(&self, outputs: &RecipeOutputs) -> bool {
        outputs
            .register
            .unwrap_or(self.recipe_type == RecipeType::Register)
    }

    fn runner_owner(&self) -> &'static str {
        match self.recipe_type {
            RecipeType::Register => "magere recipe apply (implemented here)",
            RecipeType::Goz1Pack | RecipeType::TernaryPack => {
                "issue #19 (magere pack-goz1) — not implemented here"
            }
            RecipeType::GgufExport => {
                "optional GGUF conversion placeholder — register existing GGUF without conversion"
            }
            RecipeType::Saaq => "magere run-saaq (also via magere recipe apply)",
        }
    }

    fn source_manifest_ref(&self) -> Option<&str> {
        self.inputs
            .as_ref()
            .and_then(|inputs| inputs.source_manifest.as_deref())
    }

    fn goz1_ref(&self) -> Option<&str> {
        self.inputs
            .as_ref()
            .and_then(|inputs| inputs.goz1_ref.as_deref())
    }
}

/// Same contract as `magere recipe validate`, for runners that read recipe files directly.
pub(crate) fn validate_recipe_bytes(path: &Path, contents: &str) -> Result<(), String> {
    let mut recipe =
        Recipe::from_json(contents).map_err(|e| format!("Failed to parse recipe: {e}"))?;
    recipe.source_path = Some(path.to_path_buf());
    recipe.validate()
}

/// Dispatch for `magere recipe <...>`.
pub fn run(command: RecipeCommands) -> Result<String, String> {
    match command {
        RecipeCommands::Validate { path } => validate_command(&path),
        RecipeCommands::Inspect { path } => inspect_command(&path),
        RecipeCommands::Apply { path, registry } => apply_command(&path, registry.as_deref()),
    }
}

fn validate_command(path: &Path) -> Result<String, String> {
    let recipe = Recipe::from_file(path).map_err(|e| format!("Failed to load recipe: {e}"))?;
    recipe.validate()?;
    let mut out = format!(
        "✓ Recipe '{}' is valid (type: {}, runner: {})",
        recipe.recipe_id,
        recipe.recipe_type,
        recipe.runner_owner()
    );
    for (label, resolved) in recipe.resolved_references() {
        out.push_str(&format!("\n  {label} -> {}", resolved.display()));
    }
    Ok(out)
}

fn inspect_command(path: &Path) -> Result<String, String> {
    let recipe = Recipe::from_file(path).map_err(|e| format!("Failed to load recipe: {e}"))?;
    Ok(recipe.summary())
}

fn apply_command(path: &Path, registry: Option<&Path>) -> Result<String, String> {
    let recipe = Recipe::from_file(path).map_err(|e| format!("Failed to load recipe: {e}"))?;
    recipe.apply(registry)
}

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_apply_emit;
#[cfg(test)]
mod tests_apply_register;
#[cfg(test)]
mod tests_apply_saaq;
#[cfg(test)]
mod tests_resolve;
