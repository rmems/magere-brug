//! Pack-block decoding for `magere pack-goz1`.
//!
//! The recipe document is [`crate::recipe::Recipe`]. This module only decodes the
//! `pack` object that the shared recipe stores as JSON.

use crate::recipe::{Recipe, RecipeType};
use magere_grok_process::types::InputFormat;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The `pack` block: everything needed to build a [`magere_grok_process::types::QuantizeConfig`].
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

/// Load a pack recipe through the shared recipe document and schema gate.
pub fn load_pack_recipe(recipe_path: &Path) -> Result<Recipe, String> {
    let content = std::fs::read_to_string(recipe_path)
        .map_err(|e| format!("Failed to read recipe '{}': {}", recipe_path.display(), e))?;
    let mut recipe = Recipe::from_json(&content)
        .map_err(|e| format!("Failed to parse recipe '{}': {}", recipe_path.display(), e))?;
    recipe.set_source_path(recipe_path);
    if recipe.recipe_id.trim().is_empty() {
        return Err(format!(
            "recipe '{}' is missing recipe_id",
            recipe_path.display()
        ));
    }
    recipe
        .validate_schema_document()
        .map_err(|e| format!("Failed to parse recipe '{}': {}", recipe_path.display(), e))?;
    if !matches!(
        recipe.recipe_type,
        RecipeType::Goz1Pack | RecipeType::TernaryPack
    ) {
        return Err(format!(
            "recipe '{}' has type '{}'; `pack-goz1` only runs goz1_pack or ternary_pack",
            recipe.recipe_id, recipe.recipe_type
        ));
    }
    if let Some(format) = recipe
        .outputs
        .as_ref()
        .and_then(|outputs| outputs.generated_format.as_deref())
        && format != "goz1"
    {
        return Err(format!(
            "recipe '{}' declares outputs.generated_format '{}'; `pack-goz1` only writes goz1",
            recipe.recipe_id, format
        ));
    }
    require_pack_config(&recipe)?;
    Ok(recipe)
}

pub(crate) fn require_pack_config(recipe: &Recipe) -> Result<PackConfig, String> {
    let value = recipe.pack.clone().ok_or_else(|| {
        format!(
            "recipe '{}' has no `pack` block; `pack-goz1` needs one (see configs/recipes/ternary-pack-example.json)",
            recipe.recipe_id
        )
    })?;
    serde_json::from_value(value).map_err(|e| {
        format!(
            "recipe '{}' has an invalid `pack` block: {e}",
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
