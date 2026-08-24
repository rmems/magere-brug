//! Recipe pipeline foundation.
//!
//! A recipe is a small JSON document under `configs/recipes/` that names what to
//! register, pack, or calibrate. It never carries weights; it points at manifests
//! and records the lineage of whatever the run produces.
//!
//! This module owns the recipe structure, the loader, the validator, and the
//! `register` runner. Execution of the other three types is deliberately not
//! implemented here:
//!
//! - `goz1_pack` / `ternary_pack` -> issue #19 (`magere pack-goz1`)
//! - `saaq` -> issue #8 (`magere saaq-run`)
//!
//! [`Recipe::apply`] returns an error naming the owning issue for those types,
//! which is the intended extension point for those follow-up PRs.

use crate::manifest::Manifest;
use crate::registry::ArtifactRegistry;
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Source of truth for the recipe shape, embedded so the CLI never depends on
/// the working directory to find `schemas/recipe.schema.json`.
const RECIPE_SCHEMA: &str = include_str!("../../../schemas/recipe.schema.json");

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
    /// Apply a recipe. Only `type: "register"` runs here; pack/SAAQ runners are issues #19/#8
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
    /// SAAQ validation run over a source or a registered GOZ1 pack. Runner: issue #8.
    Saaq,
}

impl RecipeType {
    fn as_str(&self) -> &'static str {
        match self {
            RecipeType::Register => "register",
            RecipeType::Goz1Pack => "goz1_pack",
            RecipeType::TernaryPack => "ternary_pack",
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
    /// Path the recipe was loaded from. Never serialized; used to resolve
    /// relative manifest references without depending on the caller's cwd.
    #[serde(skip)]
    source_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeInputs {
    /// Manifest path (`*.json`, resolved and parsed) or a registry id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_manifest: Option<String>,
    /// Asserted `source_artifact.format` of the referenced manifest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_format: Option<String>,
    /// Manifest carrying a registered GOZ1 pack, or a registry id.
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

/// Calibration config. Only meaningful for the ternary/GOZ1 pack and SAAQ paths.
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

impl Recipe {
    /// Load a recipe from a JSON string. Relative references resolve against cwd.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
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
        let instance = serde_json::to_value(self)
            .map_err(|e| format!("failed to serialize recipe for validation: {e}"))?;

        validate_against_schema(&instance)?;
        self.validate_semantics()
    }

    /// Execute the recipe.
    ///
    /// Only `type: "register"` is implemented. The other types return an error
    /// naming the issue that owns their runner.
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
            RecipeType::Saaq => Err(format!(
                "recipe '{}': saaq execution is not implemented in this command \
                 — see issue #8 (magere saaq-run)",
                self.recipe_id
            )),
        }
    }

    /// Human-readable rendering used by `magere recipe inspect`.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Recipe ID: {}\n", self.recipe_id));
        out.push_str(&format!("Type: {}\n", self.recipe_type));
        if let Some(description) = &self.description {
            out.push_str(&format!("Description: {description}\n"));
        }
        out.push_str(&format!("Runner: {}\n", self.runner_owner()));

        if let Some(inputs) = &self.inputs {
            if let Some(source_manifest) = &inputs.source_manifest {
                out.push_str(&format!("Source Manifest: {source_manifest}\n"));
            }
            if let Some(source_format) = &inputs.source_format {
                out.push_str(&format!("Source Format: {source_format}\n"));
            }
            if let Some(goz1_ref) = &inputs.goz1_ref {
                out.push_str(&format!("GOZ1 Ref: {goz1_ref}\n"));
            }
        }

        if let Some(outputs) = &self.outputs {
            if let Some(generated_format) = &outputs.generated_format {
                out.push_str(&format!("Generated Format: {generated_format}\n"));
            }
            if let Some(manifest_id) = &outputs.manifest_id {
                out.push_str(&format!("Output Manifest ID: {manifest_id}\n"));
            }
            if let Some(artifact_path) = &outputs.artifact_path {
                out.push_str(&format!("Output Artifact Path: {artifact_path}\n"));
            }
            if let Some(output_dir) = &outputs.output_dir {
                out.push_str(&format!("Output Dir: {output_dir}\n"));
            }
            if let Some(version) = outputs.goz1_version {
                out.push_str(&format!("GOZ1 Version: {version}\n"));
            }
            if let Some(algorithm) = &outputs.checksum_algorithm {
                out.push_str(&format!("Checksum Algorithm: {algorithm}\n"));
            }
            out.push_str(&format!(
                "Register Output: {}\n",
                outputs.register.unwrap_or(false)
            ));
            if let Some(registry_path) = &outputs.registry_path {
                out.push_str(&format!("Registry Path: {registry_path}\n"));
            }
            if let Some(lineage) = &outputs.lineage {
                if let Some(parent_manifest_id) = &lineage.parent_manifest_id {
                    out.push_str(&format!("Lineage Parent Manifest: {parent_manifest_id}\n"));
                }
                if let Some(parent_path) = &lineage.parent_path {
                    out.push_str(&format!("Lineage Parent Path: {parent_path}\n"));
                }
                if let Some(recipe_id) = &lineage.recipe_id {
                    out.push_str(&format!("Lineage Recipe: {recipe_id}\n"));
                }
            }
        }

        if let Some(calibration) = &self.calibration {
            out.push_str(&format!("Calibration Dataset: {}\n", calibration.dataset));
            if let Some(dataset_path) = &calibration.dataset_path {
                out.push_str(&format!("Calibration Dataset Path: {dataset_path}\n"));
            }
            if let Some(config_path) = &calibration.config_path {
                out.push_str(&format!("Calibration Config Path: {config_path}\n"));
            }
            if let Some(sample_count) = calibration.sample_count {
                out.push_str(&format!("Calibration Samples: {sample_count}\n"));
            }
            if let Some(seed) = calibration.seed {
                out.push_str(&format!("Calibration Seed: {seed}\n"));
            }
        }

        if let Some(handoff) = &self.handoff {
            for (name, target) in handoff.targets() {
                let status = target.status.as_deref().unwrap_or("placeholder");
                let enabled = target.enabled.unwrap_or(false);
                out.push_str(&format!(
                    "Handoff [{name}]: status={status}, enabled={enabled} (forward-declared; executed downstream)\n"
                ));
                if let Some(kernel_types) = &target.kernel_types {
                    out.push_str(&format!(
                        "Handoff [{name}] kernels: {}\n",
                        kernel_types.join(", ")
                    ));
                }
                if let Some(pipeline_id) = &target.pipeline_id {
                    out.push_str(&format!("Handoff [{name}] pipeline: {pipeline_id}\n"));
                }
                if let Some(notes) = &target.notes {
                    out.push_str(&format!("Handoff [{name}] notes: {notes}\n"));
                }
            }
        }

        out
    }

    /// Which command/issue owns execution of this recipe type.
    fn runner_owner(&self) -> &'static str {
        match self.recipe_type {
            RecipeType::Register => "magere recipe apply (implemented here)",
            RecipeType::Goz1Pack | RecipeType::TernaryPack => {
                "issue #19 (magere pack-goz1) — not implemented here"
            }
            RecipeType::Saaq => "issue #8 (magere saaq-run) — not implemented here",
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

    fn validate_semantics(&self) -> Result<(), String> {
        // Required references per type. The schema enforces the same rules; they
        // are repeated so recipes built in memory get identical guarantees.
        match self.recipe_type {
            RecipeType::Register | RecipeType::Goz1Pack | RecipeType::TernaryPack => {
                if self.source_manifest_ref().is_none() {
                    return Err(format!(
                        "recipe type '{}' requires inputs.source_manifest",
                        self.recipe_type
                    ));
                }
            }
            RecipeType::Saaq => {
                if self.source_manifest_ref().is_none() && self.goz1_ref().is_none() {
                    return Err(
                        "recipe type 'saaq' requires inputs.source_manifest or inputs.goz1_ref"
                            .to_string(),
                    );
                }
                // Mirrors the schema's `saaq` branch so the semantic layer stands alone.
                match self.outputs.as_ref() {
                    None => {
                        return Err("recipe type 'saaq' requires outputs".to_string());
                    }
                    Some(outputs) if outputs.output_dir.is_none() => {
                        return Err(
                            "recipe type 'saaq' requires outputs.output_dir to place the run"
                                .to_string(),
                        );
                    }
                    Some(_) => {}
                }
            }
        }

        if self.recipe_type == RecipeType::Register && self.calibration.is_some() {
            return Err(
                "calibration is only meaningful for the ternary/goz1 pack and saaq paths; \
                 a register recipe must not carry it"
                    .to_string(),
            );
        }

        if let Some(source_format) = self
            .inputs
            .as_ref()
            .and_then(|inputs| inputs.source_format.as_deref())
        {
            if !VALID_SOURCE_FORMATS.contains(&source_format) {
                return Err(format!(
                    "inputs.source_format '{source_format}' must be one of: {}",
                    VALID_SOURCE_FORMATS.join(", ")
                ));
            }
            self.reject_unpackable_source_format(source_format)?;
        }

        self.validate_outputs()?;
        self.validate_references()
    }

    /// A pack recipe naming a source the packer cannot consume passes schema
    /// validation but could never be executed by the issue #19 runner, so reject
    /// it up front rather than at pack time.
    fn reject_unpackable_source_format(&self, source_format: &str) -> Result<(), String> {
        if self.recipe_type.is_pack() && !PACKABLE_SOURCE_FORMATS.contains(&source_format) {
            return Err(format!(
                "recipe type '{}' cannot pack source format '{source_format}'; \
                 the packer accepts: {}",
                self.recipe_type,
                PACKABLE_SOURCE_FORMATS.join(", ")
            ));
        }
        Ok(())
    }

    /// Cross-check declared pack provenance against the manifest being packed so
    /// a typo cannot be persisted as false artifact lineage.
    fn validate_pack_lineage(&self, manifest: &Manifest) -> Result<(), String> {
        let Some(lineage) = self
            .outputs
            .as_ref()
            .and_then(|outputs| outputs.lineage.as_ref())
        else {
            return Ok(());
        };

        if let Some(parent_manifest_id) = lineage.parent_manifest_id.as_deref()
            && parent_manifest_id != manifest.metadata.manifest_id
        {
            return Err(format!(
                "outputs.lineage.parent_manifest_id '{parent_manifest_id}' does not match the \
                 referenced manifest's metadata.manifest_id '{}'",
                manifest.metadata.manifest_id
            ));
        }

        if let Some(parent_path) = lineage.parent_path.as_deref()
            && parent_path != manifest.source_artifact.path
        {
            return Err(format!(
                "outputs.lineage.parent_path '{parent_path}' does not match the referenced \
                 manifest's source_artifact.path '{}'",
                manifest.source_artifact.path
            ));
        }

        Ok(())
    }

    fn validate_outputs(&self) -> Result<(), String> {
        let Some(outputs) = &self.outputs else {
            return Ok(());
        };

        if let Some(generated_format) = outputs.generated_format.as_deref() {
            if !VALID_GENERATED_FORMATS.contains(&generated_format) {
                return Err(format!(
                    "outputs.generated_format '{generated_format}' must be one of: {}",
                    VALID_GENERATED_FORMATS.join(", ")
                ));
            }
            match self.recipe_type {
                RecipeType::Goz1Pack if generated_format != "goz1" => {
                    return Err(format!(
                        "a goz1_pack recipe must set outputs.generated_format to 'goz1' (got '{generated_format}')"
                    ));
                }
                RecipeType::TernaryPack
                    if generated_format != "goz1" && generated_format != "ternary" =>
                {
                    return Err(format!(
                        "a ternary_pack recipe must set outputs.generated_format to 'goz1' or 'ternary' (got '{generated_format}')"
                    ));
                }
                _ => {}
            }
        } else if self.recipe_type.is_pack() {
            return Err(format!(
                "recipe type '{}' requires outputs.generated_format",
                self.recipe_type
            ));
        }

        if let Some(version) = outputs.goz1_version {
            if version != SUPPORTED_GOZ1_VERSION {
                return Err(format!(
                    "outputs.goz1_version must be {SUPPORTED_GOZ1_VERSION} (got {version})"
                ));
            }
            if outputs.generated_format.as_deref() != Some("goz1") {
                return Err(
                    "outputs.goz1_version is only meaningful when outputs.generated_format is 'goz1'"
                        .to_string(),
                );
            }
        }

        if let Some(algorithm) = outputs.checksum_algorithm.as_deref()
            && algorithm != "sha256"
        {
            return Err(format!(
                "outputs.checksum_algorithm '{algorithm}' is not supported; use sha256"
            ));
        }

        if let Some(lineage) = &outputs.lineage
            && let Some(recipe_id) = lineage.recipe_id.as_deref()
            && recipe_id != self.recipe_id
        {
            return Err(format!(
                "outputs.lineage.recipe_id '{recipe_id}' must match recipe_id '{}'",
                self.recipe_id
            ));
        }

        if outputs.register == Some(false) && self.recipe_type == RecipeType::Register {
            return Err(
                "a register recipe exists to add its source manifest to the artifact registry, \
                 so outputs.register must not be false"
                    .to_string(),
            );
        }

        if outputs.register == Some(true) && self.recipe_type.is_pack() {
            if outputs.manifest_id.is_none() {
                return Err(
                    "outputs.register is true, so outputs.manifest_id is required to emit an artifact manifest"
                        .to_string(),
                );
            }
            if outputs.artifact_path.is_none() {
                return Err(
                    "outputs.register is true, so outputs.artifact_path is required to record the emitted artifact"
                        .to_string(),
                );
            }
        }

        Ok(())
    }

    fn validate_references(&self) -> Result<(), String> {
        if let Some(reference) = self.source_manifest_ref()
            && let Some(manifest) =
                self.load_referenced_manifest(reference, "inputs.source_manifest")?
        {
            if let Some(expected) = self
                .inputs
                .as_ref()
                .and_then(|inputs| inputs.source_format.as_deref())
                && manifest.source_artifact.format != expected
            {
                return Err(format!(
                    "inputs.source_format '{expected}' does not match the referenced manifest's \
                     source_artifact.format '{}'",
                    manifest.source_artifact.format
                ));
            }

            if self.recipe_type.is_pack() {
                // The declared `inputs.source_format` is optional, so the manifest the
                // recipe actually points at is the authoritative source format here.
                self.reject_unpackable_source_format(&manifest.source_artifact.format)?;
                self.validate_pack_lineage(&manifest)?;
            }

            if self.recipe_type == RecipeType::Register {
                if let Some(manifest_id) = self
                    .outputs
                    .as_ref()
                    .and_then(|outputs| outputs.manifest_id.as_deref())
                    && manifest_id != manifest.metadata.manifest_id
                {
                    return Err(format!(
                        "outputs.manifest_id '{manifest_id}' does not match the referenced \
                         manifest's metadata.manifest_id '{}'; a register recipe records an \
                         existing manifest, it does not rename one",
                        manifest.metadata.manifest_id
                    ));
                }

                if let Some(generated_format) = self
                    .outputs
                    .as_ref()
                    .and_then(|outputs| outputs.generated_format.as_deref())
                {
                    match manifest.generated_artifact.as_ref() {
                        Some(generated) if generated.format == generated_format => {}
                        Some(generated) => {
                            return Err(format!(
                                "outputs.generated_format '{generated_format}' does not match the \
                                 referenced manifest's generated_artifact.format '{}'",
                                generated.format
                            ));
                        }
                        None => {
                            return Err(format!(
                                "outputs.generated_format '{generated_format}' is set but the \
                                 referenced manifest has no generated_artifact to register"
                            ));
                        }
                    }
                }
            }
        }

        if let Some(reference) = self.goz1_ref()
            && let Some(manifest) = self.load_referenced_manifest(reference, "inputs.goz1_ref")?
        {
            match manifest.generated_artifact.as_ref() {
                Some(generated) if generated.format == "goz1" => {}
                Some(generated) => {
                    return Err(format!(
                        "inputs.goz1_ref must point at a manifest whose generated_artifact.format \
                         is 'goz1' (got '{}')",
                        generated.format
                    ));
                }
                None => {
                    return Err(
                        "inputs.goz1_ref must point at a manifest that carries a generated_artifact"
                            .to_string(),
                    );
                }
            }
        }

        Ok(())
    }

    /// Load a manifest reference.
    ///
    /// Every reference must name a `*.json` manifest path. Nothing in this
    /// workspace resolves registry ids yet — `apply` rejects them outright — so a
    /// non-path reference is a typo, not a feature, and is refused rather than
    /// silently skipping every cross-check below.
    fn load_referenced_manifest(
        &self,
        reference: &str,
        label: &str,
    ) -> Result<Option<Manifest>, String> {
        if !reference_is_manifest_path(reference) {
            return Err(format!(
                "{label} '{reference}' must name a manifest path ending in '.json'; \
                 registry-id references are not resolvable yet (see issues #19/#8)"
            ));
        }

        let resolved = self.resolve_reference(reference).ok_or_else(|| {
            format!("{label} '{reference}' could not be resolved to a file on disk")
        })?;

        let manifest = Manifest::from_file(&resolved).map_err(|e| {
            format!(
                "{label} '{}' is not a parseable manifest: {e}",
                resolved.display()
            )
        })?;

        manifest.validate().map_err(|e| {
            format!(
                "{label} '{}' is not a valid manifest: {e}",
                resolved.display()
            )
        })?;

        Ok(Some(manifest))
    }

    /// Resolve an input reference that must already exist on disk.
    ///
    /// Relative references are tried against the recipe file's directory and each
    /// of its ancestors (so `configs/recipes/x.json` can name a repo-root-relative
    /// `manifests/examples/y.json`), then against the working directory.
    fn resolve_reference(&self, reference: &str) -> Option<PathBuf> {
        let candidate = Path::new(reference);

        if candidate.is_absolute() {
            return candidate.is_file().then(|| candidate.to_path_buf());
        }

        if let Some(dir) = self.source_path.as_deref().and_then(Path::parent) {
            for ancestor in dir.ancestors() {
                let joined = ancestor.join(candidate);
                if joined.is_file() {
                    return Some(joined);
                }
            }
        }

        candidate.is_file().then(|| candidate.to_path_buf())
    }

    fn apply_register(&self, registry_override: Option<&Path>) -> Result<String, String> {
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

        // `outputs.register` defaults to true for a register recipe: the type exists to
        // add its source manifest to the registry. `validate_outputs` rejects an explicit
        // `false` as contradictory, so this guard only fires if a caller skips validation.
        let should_register = self
            .outputs
            .as_ref()
            .and_then(|outputs| outputs.register)
            .unwrap_or(true);
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

        // Output paths are resolved against the working directory: unlike inputs
        // they need not exist yet, so there is nothing to search for.
        let registry_path = match registry_override {
            Some(path) => path.to_path_buf(),
            None => self
                .outputs
                .as_ref()
                .and_then(|outputs| outputs.registry_path.as_deref())
                .map_or_else(|| PathBuf::from(DEFAULT_REGISTRY_PATH), PathBuf::from),
        };

        let mut registry = if registry_path.exists() {
            let content = std::fs::read_to_string(&registry_path)
                .map_err(|e| format!("failed to read registry {}: {e}", registry_path.display()))?;
            ArtifactRegistry::from_json(&content)
                .map_err(|e| format!("failed to parse registry {}: {e}", registry_path.display()))?
        } else {
            ArtifactRegistry::new()
        };

        registry.register(&manifest)?;

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

        std::fs::write(&registry_path, serialized)
            .map_err(|e| format!("failed to write registry {}: {e}", registry_path.display()))?;

        let mut out = String::new();
        out.push_str(&format!(
            "✓ Recipe '{}' ({}) applied\n",
            self.recipe_id, self.recipe_type
        ));
        out.push_str(&format!(
            "  Manifest: {} ({})\n",
            manifest_path.display(),
            manifest.metadata.manifest_id
        ));
        out.push_str(&format!(
            "  Model: {} (slug: {}, family: {})\n",
            manifest.model.name, manifest.model.slug, manifest.model.family
        ));
        out.push_str(&format!(
            "  Source: {} {}\n",
            manifest.source_artifact.format, manifest.source_artifact.path
        ));

        if let Some(generated) = &manifest.generated_artifact {
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

        out.push_str(&format!("  Registered to: {}", registry_path.display()));
        Ok(out)
    }
}

/// A reference is treated as a filesystem path when it names a JSON file;
/// anything else is a registry id resolved elsewhere.
fn reference_is_manifest_path(reference: &str) -> bool {
    reference.ends_with(".json")
}

fn validate_against_schema(instance: &Value) -> Result<(), String> {
    let schema: Value = serde_json::from_str(RECIPE_SCHEMA)
        .map_err(|e| format!("embedded recipe schema is not valid JSON: {e}"))?;

    let validator = jsonschema::draft7::new(&schema)
        .map_err(|e| format!("embedded recipe schema failed to compile: {e}"))?;

    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|error| {
            let location = error.instance_path().to_string();
            if location.is_empty() {
                format!("<root>: {error}")
            } else {
                format!("{location}: {error}")
            }
        })
        .collect();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "recipe schema validation failed:\n  - {}",
            errors.join("\n  - ")
        ))
    }
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
    Ok(format!(
        "✓ Recipe '{}' is valid (type: {}, runner: {})",
        recipe.recipe_id,
        recipe.recipe_type,
        recipe.runner_owner()
    ))
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
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .canonicalize()
            .expect("repo root is reachable from CARGO_MANIFEST_DIR")
    }

    fn sample_manifest_json(manifest_id: &str, slug: &str, source_format: &str) -> String {
        format!(
            r#"{{
  "metadata": {{
    "schema_version": 1,
    "created_at": "2026-08-23T00:00:00Z",
    "manifest_id": "{manifest_id}"
  }},
  "model": {{
    "slug": "{slug}",
    "name": "Sample Model",
    "family": "sample",
    "parameter_count": {{ "active": 1000000 }},
    "architecture": "dense"
  }},
  "source_artifact": {{
    "format": "{source_format}",
    "path": "/models/sample/model.{source_format}"
  }}
}}"#
        )
    }

    /// Writes a manifest plus a recipe into a temp dir and loads the recipe.
    fn recipe_in_temp_dir(recipe_json: &str, manifest_json: &str) -> (TempDir, Recipe) {
        let dir = TempDir::new().expect("temp dir");
        std::fs::write(dir.path().join("manifest.json"), manifest_json).expect("write manifest");
        let recipe_path = dir.path().join("recipe.json");
        std::fs::write(&recipe_path, recipe_json).expect("write recipe");
        let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
        (dir, recipe)
    }

    #[test]
    fn test_checked_in_recipes_are_valid() {
        let recipes_dir = repo_root().join("configs").join("recipes");
        let entries = std::fs::read_dir(&recipes_dir).expect("configs/recipes is readable");

        let mut checked = 0;
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let recipe = Recipe::from_file(&path)
                .unwrap_or_else(|e| panic!("{} failed to load: {e}", path.display()));
            recipe
                .validate()
                .unwrap_or_else(|e| panic!("{} failed to validate: {e}", path.display()));
            checked += 1;
        }

        assert!(
            checked >= 5,
            "expected all five register + pack recipe examples to be present, found {checked}"
        );
    }

    #[test]
    fn test_valid_register_recipe() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "sample-register",
              "type": "register",
              "description": "register a safetensors source",
              "inputs": {
                "source_manifest": "manifest.json",
                "source_format": "safetensors"
              },
              "outputs": { "register": true, "manifest_id": "sample-v1" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );

        assert!(recipe.validate().is_ok(), "{:?}", recipe.validate());
        assert_eq!(recipe.recipe_type, RecipeType::Register);
    }

    #[test]
    fn test_register_requires_source_manifest() {
        let recipe = Recipe::from_json(r#"{ "recipe_id": "no-inputs", "type": "register" }"#)
            .expect("parses");
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("inputs"), "unexpected error: {err}");
    }

    #[test]
    fn test_pack_requires_source_manifest_and_generated_format() {
        let recipe = Recipe::from_json(
            r#"{
              "recipe_id": "pack-no-inputs",
              "type": "goz1_pack",
              "outputs": { "generated_format": "goz1" }
            }"#,
        )
        .expect("parses");
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("inputs"), "unexpected error: {err}");

        let recipe = Recipe::from_json(
            r#"{
              "recipe_id": "pack-no-source-manifest",
              "type": "goz1_pack",
              "inputs": { "source_format": "safetensors" },
              "outputs": { "generated_format": "goz1" }
            }"#,
        )
        .expect("parses");
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("source_manifest"), "unexpected error: {err}");

        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "pack-no-outputs",
              "type": "ternary_pack",
              "inputs": { "source_manifest": "manifest.json" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("outputs"), "unexpected error: {err}");
    }

    #[test]
    fn test_saaq_requires_source_manifest_or_goz1_ref() {
        let recipe = Recipe::from_json(
            r#"{
              "recipe_id": "saaq-no-inputs",
              "type": "saaq",
              "outputs": { "output_dir": "/runs/saaq" }
            }"#,
        )
        .expect("parses");
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("inputs"), "unexpected error: {err}");

        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "saaq-ok",
              "type": "saaq",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": { "output_dir": "/runs/saaq" },
              "calibration": { "dataset": "wikitext-2", "sample_count": 128, "seed": 7 }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );
        assert!(recipe.validate().is_ok(), "{:?}", recipe.validate());
    }

    #[test]
    fn test_calibration_rejected_on_register_recipe() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "register-with-calibration",
              "type": "register",
              "inputs": { "source_manifest": "manifest.json" },
              "calibration": { "dataset": "wikitext-2" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("calibration"), "unexpected error: {err}");
    }

    #[test]
    fn test_awq_and_gptq_are_rejected() {
        // The ban lives in the schema's `safe_string` pattern, so it must fire on a
        // free-text field and report a schema failure, not merely echo the input.
        for removed in ["awq", "AWQ", "gptq", "GPTQ"] {
            let json = format!(
                r#"{{
                  "recipe_id": "sample-register",
                  "type": "register",
                  "description": "calibrated with the {removed} path",
                  "inputs": {{ "source_manifest": "manifest.json" }}
                }}"#
            );
            let (_dir, recipe) = recipe_in_temp_dir(
                &json,
                &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
            );
            let err = recipe
                .validate()
                .expect_err("removed quantization paths must be rejected");
            assert!(
                err.contains("recipe schema validation failed"),
                "expected a schema rejection for '{removed}', got: {err}"
            );
            assert!(err.contains("/description"), "unexpected error: {err}");
        }
    }

    #[test]
    fn test_unresolvable_manifest_reference_is_rejected() {
        let dir = TempDir::new().expect("temp dir");
        let recipe_path = dir.path().join("recipe.json");
        std::fs::write(
            &recipe_path,
            r#"{
              "recipe_id": "dangling-ref",
              "type": "register",
              "inputs": { "source_manifest": "does-not-exist.json" }
            }"#,
        )
        .expect("write recipe");
        let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("could not be resolved"), "unexpected: {err}");
    }

    #[test]
    fn test_source_format_mismatch_is_rejected() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "format-mismatch",
              "type": "register",
              "inputs": { "source_manifest": "manifest.json", "source_format": "gguf" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("source_format"), "unexpected error: {err}");
    }

    #[test]
    fn test_manifest_id_mismatch_is_rejected() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "id-mismatch",
              "type": "register",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": { "manifest_id": "some-other-id" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("manifest_id"), "unexpected error: {err}");
    }

    #[test]
    fn test_goz1_version_must_match_writer() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "bad-goz1-version",
              "type": "goz1_pack",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": { "generated_format": "goz1", "goz1_version": 2 }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("goz1_version"), "unexpected error: {err}");
    }

    #[test]
    fn test_goz1_pack_generated_format_is_constrained() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "wrong-generated-format",
              "type": "goz1_pack",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": { "generated_format": "gguf" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("generated_format"), "unexpected error: {err}");
    }

    #[test]
    fn test_pack_register_requires_manifest_id_and_artifact_path() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "pack-register-incomplete",
              "type": "ternary_pack",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": { "generated_format": "goz1", "register": true }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );
        let err = recipe.validate().expect_err("must be rejected");
        assert!(
            err.contains("manifest_id") || err.contains("artifact_path"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_unknown_field_is_rejected() {
        let err =
            Recipe::from_json(r#"{ "recipe_id": "extra", "type": "register", "not_a_field": 1 }"#)
                .expect_err("unknown fields must be rejected");
        assert!(err.to_string().contains("not_a_field"), "{err}");
    }

    #[test]
    fn test_lineage_recipe_id_must_match() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "lineage-owner",
              "type": "goz1_pack",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": {
                "generated_format": "goz1",
                "lineage": { "recipe_id": "someone-else" }
              }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("lineage"), "unexpected error: {err}");
    }

    #[test]
    fn test_pack_recipe_rejects_unpackable_source_format() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "pack-gguf-source",
              "type": "ternary_pack",
              "inputs": { "source_manifest": "manifest.json", "source_format": "gguf" },
              "outputs": { "generated_format": "goz1" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "gguf"),
        );

        // Encoded twice on purpose: the schema is the source of truth and fires
        // first, and the semantic layer must stand alone for callers that skip it.
        let err = recipe
            .validate()
            .expect_err("the packer cannot consume gguf, so a pack recipe must not declare it");
        assert!(err.contains("/inputs/source_format"), "{err}");

        let err = recipe
            .validate_semantics()
            .expect_err("the semantic layer must reject it too");
        assert!(err.contains("cannot pack source format 'gguf'"), "{err}");
    }

    #[test]
    fn test_register_recipe_accepts_gguf_source_format() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "register-gguf-source",
              "type": "register",
              "inputs": { "source_manifest": "manifest.json", "source_format": "gguf" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "gguf"),
        );

        recipe
            .validate()
            .expect("gguf stays a valid registry source format for register recipes");
    }

    #[test]
    fn test_pack_lineage_parent_manifest_id_must_match_source() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "pack-bad-lineage-id",
              "type": "ternary_pack",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": {
                "generated_format": "goz1",
                "lineage": { "parent_manifest_id": "typo-v9" }
              }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );

        let err = recipe
            .validate()
            .expect_err("mismatched pack lineage must not validate");
        assert!(
            err.contains("outputs.lineage.parent_manifest_id 'typo-v9'"),
            "{err}"
        );
    }

    #[test]
    fn test_pack_lineage_parent_path_must_match_source() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "pack-bad-lineage-path",
              "type": "ternary_pack",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": {
                "generated_format": "goz1",
                "lineage": { "parent_path": "/models/wrong/path" }
              }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );

        let err = recipe
            .validate()
            .expect_err("mismatched pack lineage path must not validate");
        assert!(
            err.contains("outputs.lineage.parent_path '/models/wrong/path'"),
            "{err}"
        );
    }

    #[test]
    fn test_register_recipe_rejects_register_false() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "register-disabled",
              "type": "register",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": { "register": false }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );

        let err = recipe
            .validate()
            .expect_err("outputs.register false must be rejected on a register recipe");
        assert!(err.contains("/outputs/register"), "{err}");

        let err = recipe
            .validate_semantics()
            .expect_err("the semantic layer must reject it too");
        assert!(err.contains("outputs.register must not be false"), "{err}");
    }

    #[test]
    fn test_apply_register_false_writes_no_registry() {
        let (dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "register-disabled-apply",
              "type": "register",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": { "register": false }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );

        let registry_path = dir.path().join("registry.json");
        recipe
            .apply(Some(&registry_path))
            .expect_err("a register recipe declaring register:false must not apply");
        assert!(
            !registry_path.exists(),
            "registry was written despite outputs.register being false"
        );
    }

    #[test]
    fn test_apply_register_writes_registry() {
        let (dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "sample-register-apply",
              "type": "register",
              "inputs": { "source_manifest": "manifest.json" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );

        let registry_path = dir.path().join("nested").join("registry.json");
        let output = recipe
            .apply(Some(&registry_path))
            .expect("register recipe applies");

        assert!(output.contains("sample-register-apply"), "{output}");
        assert!(registry_path.is_file(), "registry file was not written");

        let written = std::fs::read_to_string(&registry_path).expect("read registry");
        let registry = ArtifactRegistry::from_json(&written).expect("parse registry");
        assert!(registry.models.contains_key("sample_model"));
    }

    #[test]
    fn test_apply_pack_types_name_owning_issue() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "pack-not-implemented",
              "type": "goz1_pack",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": { "generated_format": "goz1" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );
        let err = recipe.apply(None).expect_err("must not execute");
        assert!(err.contains("#19"), "unexpected error: {err}");
    }

    #[test]
    fn test_apply_saaq_names_owning_issue() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "saaq-not-implemented",
              "type": "saaq",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": { "output_dir": "/runs/saaq" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );
        let err = recipe.apply(None).expect_err("must not execute");
        assert!(err.contains("#8"), "unexpected error: {err}");
    }

    #[test]
    fn test_summary_reports_type_and_runner_owner() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "summary-check",
              "type": "ternary_pack",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": { "generated_format": "goz1", "goz1_version": 1 },
              "calibration": { "dataset": "wikitext-2", "sample_count": 64, "seed": 0 },
              "handoff": {
                "myelin_accelerator": {
                  "enabled": false,
                  "status": "placeholder",
                  "kernel_types": ["ternary", "saaq"]
                }
              }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );

        let summary = recipe.summary();
        assert!(summary.contains("ternary_pack"), "{summary}");
        assert!(summary.contains("#19"), "{summary}");
        assert!(summary.contains("wikitext-2"), "{summary}");
        assert!(summary.contains("myelin_accelerator"), "{summary}");
    }

    #[test]
    fn test_goz1_ref_must_point_at_a_goz1_manifest() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "bad-goz1-ref",
              "type": "saaq",
              "inputs": { "goz1_ref": "manifest.json" },
              "outputs": { "output_dir": "/runs/saaq" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );
        let err = recipe.validate().expect_err("must be rejected");
        assert!(err.contains("goz1_ref"), "unexpected error: {err}");
    }

    #[test]
    fn test_registry_id_reference_is_rejected() {
        let recipe = Recipe::from_json(
            r#"{
              "recipe_id": "registry-id-ref",
              "type": "register",
              "inputs": { "source_manifest": "olmoe_baseline" }
            }"#,
        )
        .expect("parses");

        let err = recipe
            .validate()
            .expect_err("registry ids are not resolvable yet, so they must not validate");
        assert!(err.contains("must name a manifest path"), "{err}");

        let err = recipe.apply(None).expect_err("apply needs a manifest path");
        assert!(err.contains("must name a manifest path"), "{err}");
    }

    #[test]
    fn test_misspelled_manifest_extension_is_rejected() {
        // The whole point of rejecting non-path references: a typo used to validate
        // clean because every cross-check was skipped.
        for reference in ["manifest.jsonn", "manifest.JSON", "manifest.yaml"] {
            let (_dir, recipe) = recipe_in_temp_dir(
                &format!(
                    r#"{{
                      "recipe_id": "typo-ref",
                      "type": "register",
                      "inputs": {{ "source_manifest": "{reference}" }}
                    }}"#
                ),
                &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
            );
            let err = recipe
                .validate()
                .expect_err("a mistyped manifest reference must not validate");
            assert!(err.contains("must name a manifest path"), "{err}");
        }
    }

    #[test]
    fn test_pack_rejects_unpackable_resolved_manifest_format() {
        // No `inputs.source_format` declared: the resolved manifest is authoritative.
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "pack-gguf-manifest",
              "type": "goz1_pack",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": { "generated_format": "goz1" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "gguf"),
        );

        let err = recipe
            .validate()
            .expect_err("a pack recipe over a gguf manifest must not validate");
        assert!(err.contains("cannot pack source format 'gguf'"), "{err}");
    }

    #[test]
    fn test_saaq_requires_outputs_with_output_dir() {
        let (_dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "saaq-no-output-dir",
              "type": "saaq",
              "inputs": { "source_manifest": "manifest.json" },
              "outputs": { "checksum_algorithm": "sha256" }
            }"#,
            &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
        );

        let err = recipe
            .validate()
            .expect_err("a saaq recipe must declare where the run lands");
        assert!(err.contains("output_dir"), "{err}");
    }
}
