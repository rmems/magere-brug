//! Recipe document types and load/validate for `magere pack-goz1`.

use magere_grok_process::types::InputFormat;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Recipe `type` values this runner accepts.
pub(super) const PACK_RECIPE_TYPES: &[&str] = &["goz1_pack", "ternary_pack"];

/// A recipe document as consumed by `magere pack-goz1`.
///
/// Strict like the inner blocks: a misspelled `outputs` or `inputs` would otherwise be
/// ignored while this runner silently applies its defaults and registers an artifact under
/// a name the recipe never asked for. The sibling `saaq` block is declared (and ignored) so
/// a recipe file that declares both `pack` and `saaq` still loads; a future runner block
/// must be added here alongside its schema entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Runner block owned by `magere run-saaq`; parsed so mixed recipes load, never read here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saaq: Option<serde_json::Value>,
}

/// Strict: the schema declares `additionalProperties: false` for this block, and the CLI does
/// not schema-validate recipes, so serde is the only thing enforcing that contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeInputs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_manifest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goz1_ref: Option<String>,
}

/// Strict, for the same reason as [`RecipeInputs`] -- and one sharper one: every field here
/// has a silent fallback. A misspelled `manifest_ids` would otherwise be ignored and the run
/// would quietly adopt the default identity `<source manifest id>-goz1`, writing and
/// registering an artifact under a name the recipe never asked for.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Skip the family check between the source manifest and the dissect document.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_model_mismatch: Option<bool>,
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

pub(super) fn require_pack_config(recipe: &PackRecipe) -> Result<&PackConfig, String> {
    recipe.pack.as_ref().ok_or_else(|| {
        format!(
            "recipe '{}' has no `pack` block; `pack-goz1` needs one (see configs/recipes/ternary-pack-example.json)",
            recipe.recipe_id
        )
    })
}

/// Map a manifest `source_artifact.format` onto a packer input format.
///
/// GGUF and `hf_repo` are registry source formats but not packer inputs; such recipes must
/// name `pack.input_format` explicitly.
pub(super) fn input_format_for(source_format: &str) -> Result<InputFormat, String> {
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
