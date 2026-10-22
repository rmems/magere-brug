use super::super::emit::format_outcome;
use super::harness::*;
use crate::checksum;
use crate::pack::*;
use crate::registry::ArtifactRegistry;
use magere_grok_process::types::InputFormat;
use serde_json::json;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

#[test]
fn end_to_end_writes_parseable_goz1_and_registers_it() {
    let h = default_harness();
    let outcome = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).expect("pack run");

    let bytes = fs::read(&outcome.pack_path).unwrap();
    assert!(bytes.starts_with(b"GOZ1"), "pack must start with GOZ1");
    assert_fixture_counts(&outcome, &bytes);
    assert_eq!(
        outcome.pack_path,
        h.output_dir.join("test-pack-v1.goz1"),
        "pack file name derives from outputs.manifest_id"
    );
    assert!(
        checksum::verify_checksum(&outcome.pack_path, &outcome.sha256).unwrap(),
        "recorded sha256 must match the written file"
    );
    assert_skeleton_manifest(&outcome);

    let reloaded = Manifest::from_file(&outcome.manifest_path).expect("reload manifest");
    assert!(reloaded.validate().is_ok());
    assert_eq!(reloaded.metadata.manifest_id, "test-pack-v1");
    assert_eq!(reloaded.model.slug, "grok_1_future_goz1");

    let registry =
        ArtifactRegistry::from_json(&fs::read_to_string(&outcome.registry_path).unwrap()).unwrap();
    let entry = registry
        .lookup("grok_1_future_goz1")
        .expect("registry entry");
    assert_eq!(entry.manifest_id, "test-pack-v1");
    assert!(!outcome.registry_replaced);

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
fn a_foreign_registry_entry_is_not_displaced_by_a_new_pack() {
    let h = default_harness();
    let first = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap();
    assert!(!first.registry_replaced);
    let slug = first.manifest.model.slug.clone();

    let mut variant = base_recipe(&h.output_dir);
    variant["recipe_id"] = json!("test-ternary-pack-variant");
    variant["outputs"]["manifest_id"] = json!("test-pack-v2");
    let variant_path = write_recipe(h.dir.path(), "recipe-variant.json", &variant);

    let err = run_pack_recipe(&variant_path, Some(&h.registry_path), None).unwrap_err();
    assert!(err.contains("already registered"), "{}", err);
    assert!(err.contains("test-pack-v2.goz1"), "{}", err);
    assert!(err.contains("test-pack-v2.manifest.json"), "{}", err);
    assert!(err.contains("<source model slug>_goz1"), "{}", err);
    assert!(
        !err.contains("set a different `outputs.manifest_id`"),
        "{}",
        err
    );
    assert!(h.output_dir.join("test-pack-v2.goz1").exists());

    let registry =
        ArtifactRegistry::from_json(&fs::read_to_string(&h.registry_path).unwrap()).unwrap();
    assert_eq!(registry.count(), 1);
    assert_eq!(
        registry.models.get(&slug).unwrap().manifest_id,
        "test-pack-v1"
    );
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

#[test]
fn shipped_example_recipe_is_well_formed_and_runnable() {
    let recipe = load_pack_recipe(Path::new(EXAMPLE_RECIPE)).expect("example recipe parses");
    assert_eq!(recipe.recipe_type, "ternary_pack");
    assert_eq!(
        recipe.pack.as_ref().unwrap().input_format,
        Some(InputFormat::NpyDir)
    );
    let (source_manifest, dissect, source_family, dissect_family) = shipped_example_paths();
    assert_eq!(source_family, dissect_family);
    let dir = TempDir::new().unwrap();
    let mut as_absolute: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(EXAMPLE_RECIPE).unwrap()).unwrap();
    as_absolute["inputs"]["source_manifest"] = json!(source_manifest.display().to_string());
    as_absolute["pack"]["dissect_manifest"] = json!(dissect.display().to_string());
    let recipe_path = write_recipe(dir.path(), "example.json", &as_absolute);
    let outcome = run_pack_recipe(
        &recipe_path,
        Some(&dir.path().join("registry.json")),
        Some(dir.path()),
    )
    .expect("shipped example recipe must run");
    assert!(fs::read(&outcome.pack_path).unwrap().starts_with(b"GOZ1"));
    assert_eq!(outcome.tensor_count, FIXTURE_TENSORS);
    assert_eq!(
        outcome.pack_path,
        dir.path().join("grok-1-goz1-pack-example-v1.goz1")
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
fn padded_manifest_id_agrees_between_filename_and_manifest() {
    let dir = TempDir::new().unwrap();
    let packs = dir.path().join("packs");
    let mut recipe = base_recipe(&packs);
    recipe["outputs"]["manifest_id"] = json!("  padded-pack-v1\t");
    let h = harness_from(dir, &recipe);
    let outcome = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap();
    assert_eq!(
        outcome.pack_path.file_name().unwrap(),
        "padded-pack-v1.goz1"
    );
    assert_eq!(outcome.manifest.metadata.manifest_id, "padded-pack-v1");
    let reloaded = Manifest::from_file(&outcome.manifest_path).expect("reload manifest");
    assert!(reloaded.validate().is_ok());
    assert_eq!(reloaded.metadata.manifest_id, "padded-pack-v1");
}

#[test]
fn a_mixed_pack_and_saaq_recipe_still_loads() {
    let dir = TempDir::new().unwrap();
    let mut recipe = base_recipe(&dir.path().join("packs"));
    recipe["saaq"] = json!({ "snn_steps": 4 });
    let h = harness_from(dir, &recipe);
    let parsed = load_pack_recipe(&h.recipe_path).expect("mixed recipe loads");
    assert!(parsed.saaq.is_some());
}

#[test]
fn records_the_dissect_manifest_in_source_lineage() {
    let h = default_harness();
    let outcome = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).expect("pack run");
    let lineage = outcome
        .manifest
        .generated_artifact
        .as_ref()
        .unwrap()
        .source_lineage
        .as_ref()
        .unwrap();
    assert_eq!(
        lineage.dissect_manifest_path.as_deref(),
        Some(FIXTURE_DISSECT)
    );
    let expected = checksum::compute_file_sha256(Path::new(FIXTURE_DISSECT)).unwrap();
    assert_eq!(
        lineage
            .dissect_manifest_checksum
            .as_ref()
            .unwrap()
            .sha256
            .as_deref(),
        Some(expected.as_str())
    );
    assert_eq!(lineage.path.as_deref(), Some("/models/grok-1"));
    assert!(lineage.checksum.is_none());
    let quantization = outcome.manifest.quantization.as_ref().unwrap();
    assert!(quantization.calibration_dataset.is_none());
    assert!(quantization.calibration_config_path.is_none());
}

#[test]
fn lineage_keeps_source_checksum_only_when_input_dir_is_the_source_path() {
    let dir = TempDir::new().unwrap();
    let mut recipe = base_recipe(&dir.path().join("packs"));
    recipe["pack"].as_object_mut().unwrap().remove("input_dir");
    let h = harness_from(dir, &recipe);
    let outcome = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap();
    let lineage = outcome
        .manifest
        .generated_artifact
        .as_ref()
        .unwrap()
        .source_lineage
        .as_ref()
        .unwrap();
    assert_eq!(
        lineage.path.as_deref(),
        Some("/models/grok/grok-1-checkpoint")
    );
}

#[test]
fn pack_goz1_command_renders_a_report() {
    let h = default_harness();
    let report = pack_goz1_command(&h.recipe_path, Some(&h.registry_path), None).unwrap();
    assert!(report.contains("GOZ1 pack written"));
    assert!(report.contains("generated_artifact.status: planned"));
    assert!(report.contains("SKELETON PACK"));
}

#[test]
fn allow_model_mismatch_opts_out_of_the_family_check() {
    let dir = TempDir::new().unwrap();
    let mut recipe = base_recipe(&dir.path().join("packs"));
    recipe["inputs"]["source_manifest"] = json!(REDPAJAMA_MANIFEST);
    recipe["pack"]["allow_model_mismatch"] = json!(true);
    let h = harness_from(dir, &recipe);
    let outcome = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap();
    assert_eq!(outcome.manifest.model.family, "redpajama");
}

#[test]
fn source_manifest_can_be_a_registry_slug() {
    let dir = TempDir::new().unwrap();
    let mut registry = ArtifactRegistry::new();
    let source = Manifest::from_file(SOURCE_MANIFEST).unwrap();
    registry
        .register_at(&source, Some(Path::new(SOURCE_MANIFEST)))
        .unwrap();
    let registry_path = dir.path().join("registry.json");
    fs::write(&registry_path, registry.to_json_pretty().unwrap()).unwrap();

    let mut recipe = base_recipe(&dir.path().join("packs"));
    recipe["inputs"]["source_manifest"] = json!("grok_1_future");
    let recipe_path = write_recipe(dir.path(), "by-slug.json", &recipe);
    let outcome = run_pack_recipe(&recipe_path, Some(&registry_path), None).unwrap();
    assert_eq!(outcome.manifest.model.slug, "grok_1_future_goz1");
}
