use super::harness::*;
use crate::pack::recipe::input_format_for;
use crate::pack::*;
use magere_grok_process::types::InputFormat;
use serde_json::json;
use std::fs;
use tempfile::TempDir;

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
fn rejects_a_misspelled_outputs_key() {
    let dir = TempDir::new().unwrap();
    let mut recipe = base_recipe(&dir.path().join("packs"));
    let manifest_id = recipe["outputs"]["manifest_id"].take();
    recipe["outputs"]["manifest_ids"] = manifest_id;
    let h = harness_from(dir, &recipe);
    let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
    assert!(err.contains("manifest_ids"), "{}", err);
    assert!(!h.output_dir.exists(), "nothing may be written");
}

#[test]
fn rejects_a_misspelled_inputs_key() {
    let dir = TempDir::new().unwrap();
    let mut recipe = base_recipe(&dir.path().join("packs"));
    let src = recipe["inputs"]["source_manifest"].take();
    recipe["inputs"]["source_manifests"] = src;
    let h = harness_from(dir, &recipe);
    let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
    assert!(err.contains("source_manifests"), "{}", err);
}

#[test]
fn rejects_an_unknown_top_level_recipe_key() {
    let dir = TempDir::new().unwrap();
    let mut recipe = base_recipe(&dir.path().join("packs"));
    recipe["output"] = recipe["outputs"].take();
    let h = harness_from(dir, &recipe);
    let err = load_pack_recipe(&h.recipe_path).unwrap_err();
    assert!(err.contains("output"), "{}", err);
}

#[test]
fn refuses_to_overwrite_an_input_aliased_by_an_output_path() {
    let dir = TempDir::new().unwrap();
    let output_dir = dir.path().join("packs");
    fs::create_dir_all(&output_dir).unwrap();
    let aliased_source = output_dir.join("test-pack-v1.manifest.json");
    fs::copy(SOURCE_MANIFEST, &aliased_source).unwrap();
    let mut recipe = base_recipe(&output_dir);
    recipe["inputs"]["source_manifest"] = json!(aliased_source.display().to_string());
    let h = harness_from(dir, &recipe);
    let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
    assert!(err.contains("would overwrite recipe input"), "{}", err);
    assert_eq!(
        fs::read(&aliased_source).unwrap(),
        fs::read(SOURCE_MANIFEST).unwrap(),
        "the aliased input must be left untouched"
    );
}

#[test]
fn refuses_to_use_the_registry_path_as_a_generated_output() {
    let dir = TempDir::new().unwrap();
    let output_dir = dir.path().join("packs");
    let recipe = base_recipe(&output_dir);
    let h = harness_from(dir, &recipe);
    let registry_as_pack = h.output_dir.join("test-pack-v1.goz1");
    let err = run_pack_recipe(&h.recipe_path, Some(&registry_as_pack), None).unwrap_err();
    assert!(err.contains("would overwrite"), "{}", err);
    assert!(!registry_as_pack.exists() || fs::metadata(&registry_as_pack).unwrap().len() == 0);
}

#[test]
fn a_malformed_registry_fails_before_anything_is_written() {
    let h = default_harness();
    fs::write(&h.registry_path, "{ not json").unwrap();
    let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
    assert!(err.contains("registry"), "{}", err);
    assert!(
        !h.output_dir.exists(),
        "no pack or manifest may be written when the registry is unusable"
    );
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
        "path": "/models/grok/grok-1-checkpoint"
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

    recipe["pack"]["input_format"] = json!("npy_dir");
    let recipe_path = write_recipe(dir.path(), "gguf-explicit.json", &recipe);
    let outcome = run_pack_recipe(&recipe_path, Some(&registry_path), None).unwrap();
    assert_eq!(outcome.tensor_count, FIXTURE_TENSORS);
}

#[test]
fn rejects_a_dissect_manifest_for_a_different_model_family() {
    let dir = TempDir::new().unwrap();
    let mut recipe = base_recipe(&dir.path().join("packs"));
    recipe["inputs"]["source_manifest"] = json!(REDPAJAMA_MANIFEST);
    let h = harness_from(dir, &recipe);
    let err = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap_err();
    assert!(err.contains("does not match"), "{}", err);
    assert!(err.contains("redpajama"), "{}", err);
    assert!(err.contains("grok"), "{}", err);
}
