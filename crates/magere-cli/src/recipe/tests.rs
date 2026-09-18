use super::*;
use std::path::{Path, PathBuf};
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
        checked >= 7,
        "expected register, pack, export, and saaq recipe examples to be present, found {checked}"
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
    let recipe =
        Recipe::from_json(r#"{ "recipe_id": "no-inputs", "type": "register" }"#).expect("parses");
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
          "outputs": { "output_dir": "/runs/saaq" }
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
fn test_summary_register_default_matches_apply() {
    let (dir, recipe) = recipe_in_temp_dir(
        r#"{
          "recipe_id": "register-implicit",
          "type": "register",
          "inputs": { "source_manifest": "manifest.json" },
          "outputs": { "checksum_algorithm": "sha256" }
        }"#,
        &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    );

    // inspect must not claim the registry is left alone when apply will write it.
    assert!(
        recipe.summary().contains("Register Output: true"),
        "{}",
        recipe.summary()
    );

    let registry_path = dir.path().join("registry.json");
    recipe
        .apply(Some(&registry_path))
        .expect("register applies");
    assert!(registry_path.is_file(), "apply disagreed with inspect");
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
