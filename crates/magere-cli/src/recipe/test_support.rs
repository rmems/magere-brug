use super::Recipe;
use crate::manifest::Manifest;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repo root is reachable from CARGO_MANIFEST_DIR")
}

pub fn sample_manifest_json(manifest_id: &str, slug: &str, source_format: &str) -> String {
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
"path": "model.{source_format}"
  }}
}}"#
    )
}

pub fn write_manifest_with_artifact(path: &Path, manifest_json: &str) {
    std::fs::write(path, manifest_json).expect("write manifest");
    let manifest = Manifest::from_json(manifest_json).expect("parse fixture manifest");
    if matches!(
        manifest.source_artifact.format.as_str(),
        "safetensors" | "gguf"
    ) {
        let artifact = path.parent().unwrap().join(&manifest.source_artifact.path);
        std::fs::write(artifact, b"fixture artifact").expect("write source artifact");
    }
}

/// Writes a manifest plus a recipe into a temp dir and loads the recipe.
pub fn recipe_in_temp_dir(recipe_json: &str, manifest_json: &str) -> (TempDir, Recipe) {
    let dir = TempDir::new().expect("temp dir");
    write_manifest_with_artifact(&dir.path().join("manifest.json"), manifest_json);
    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(&recipe_path, recipe_json).expect("write recipe");
    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    (dir, recipe)
}
