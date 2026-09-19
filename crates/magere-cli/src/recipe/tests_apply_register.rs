use super::test_support::{recipe_in_temp_dir, sample_manifest_json, write_manifest_with_artifact};
use super::*;
use crate::registry::ArtifactRegistry;

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
    assert!(output.contains("Emitted manifest"), "{output}");
    assert!(output.contains("combine-for-AI handoff"), "{output}");
    assert!(registry_path.is_file(), "registry file was not written");

    let written = std::fs::read_to_string(&registry_path).expect("read registry");
    let registry = ArtifactRegistry::from_json(&written).expect("parse registry");
    assert!(registry.models.contains_key("sample_model"));

    let emitted_manifest = dir
        .path()
        .join("nested")
        .join("manifests")
        .join("sample-v1.json");
    let emitted_handoff = dir
        .path()
        .join("nested")
        .join("handoff")
        .join("sample-v1.json");
    assert!(
        emitted_manifest.is_file(),
        "artifact manifest was not emitted"
    );
    assert!(
        emitted_handoff.is_file(),
        "combine-for-AI handoff was not emitted"
    );

    let handoff: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&emitted_handoff).unwrap()).unwrap();
    assert_eq!(handoff["model_slug"], "sample_model");
    assert_eq!(handoff["benchmark_linkage"]["status"], "ready");
    assert_eq!(handoff["schema"], "magere-brug/combine-for-ai-handoff/1");
}

#[test]
fn test_apply_register_is_idempotent() {
    let (dir, recipe) = recipe_in_temp_dir(
        r#"{
          "recipe_id": "sample-register-again",
          "type": "register",
          "inputs": { "source_manifest": "manifest.json" }
        }"#,
        &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    );

    let registry_path = dir.path().join("registry.json");
    recipe
        .apply(Some(&registry_path))
        .expect("first register applies");
    recipe
        .apply(Some(&registry_path))
        .expect("second register must replace rather than fail");
}

#[test]
fn test_apply_rejects_manifest_id_owned_by_another_slug_before_emitting() {
    let (dir, recipe) = recipe_in_temp_dir(
        r#"{
          "recipe_id": "manifest-owner",
          "type": "register",
          "inputs": { "source_manifest": "manifest.json" }
        }"#,
        &sample_manifest_json("shared-v1", "first_model", "safetensors"),
    );
    let registry_path = dir.path().join("registry.json");
    recipe.apply(Some(&registry_path)).expect("first apply");
    let emitted_path = dir.path().join("manifests").join("shared-v1.json");
    let handoff_path = dir.path().join("handoff").join("shared-v1.json");
    let emitted_before = std::fs::read(&emitted_path).unwrap();
    let handoff_before = std::fs::read(&handoff_path).unwrap();

    write_manifest_with_artifact(
        &dir.path().join("manifest.json"),
        &sample_manifest_json("shared-v1", "other_model", "safetensors"),
    );
    let error = recipe
        .apply(Some(&registry_path))
        .expect_err("another slug must not claim an existing manifest id");
    assert!(
        error.contains("already registered to model slug 'first_model'"),
        "{error}"
    );
    assert_eq!(std::fs::read(emitted_path).unwrap(), emitted_before);
    assert_eq!(std::fs::read(handoff_path).unwrap(), handoff_before);
}

#[test]
fn test_apply_allows_same_slug_to_update_its_manifest() {
    let (dir, recipe) = recipe_in_temp_dir(
        r#"{
          "recipe_id": "manifest-update",
          "type": "register",
          "inputs": { "source_manifest": "manifest.json" }
        }"#,
        &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    );
    let registry_path = dir.path().join("registry.json");
    recipe.apply(Some(&registry_path)).expect("first apply");

    write_manifest_with_artifact(
        &dir.path().join("manifest.json"),
        &sample_manifest_json("sample-v2", "sample_model", "safetensors"),
    );
    recipe
        .apply(Some(&registry_path))
        .expect("the owning slug may update its manifest");
    let registry =
        ArtifactRegistry::from_json(&std::fs::read_to_string(registry_path).unwrap()).unwrap();
    assert_eq!(
        registry.lookup("sample_model").unwrap().manifest_id,
        "sample-v2"
    );
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
fn test_apply_gguf_export_is_placeholder() {
    let (_dir, recipe) = recipe_in_temp_dir(
        r#"{
          "recipe_id": "gguf-export-placeholder",
          "type": "gguf_export",
          "inputs": { "source_manifest": "manifest.json", "source_format": "safetensors" },
          "outputs": { "generated_format": "gguf" }
        }"#,
        &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    );
    assert!(recipe.validate().is_ok(), "{:?}", recipe.validate());
    let err = recipe.apply(None).expect_err("must not execute conversion");
    assert!(err.contains("placeholder"), "unexpected error: {err}");
    assert!(err.contains("register"), "unexpected error: {err}");
}

#[test]
fn test_gguf_export_rejects_existing_gguf_source() {
    let (_dir, recipe) = recipe_in_temp_dir(
        r#"{
          "recipe_id": "gguf-export-from-gguf",
          "type": "gguf_export",
          "inputs": { "source_manifest": "manifest.json", "source_format": "gguf" },
          "outputs": { "generated_format": "gguf" }
        }"#,
        &sample_manifest_json("sample-v1", "sample_model", "gguf"),
    );
    let err = recipe
        .validate()
        .expect_err("existing GGUF must be registered, not re-exported");
    assert!(
        err.contains("/inputs/source_format") || err.contains("cannot convert"),
        "{err}"
    );
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
