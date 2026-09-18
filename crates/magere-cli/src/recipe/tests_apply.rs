use super::*;
use crate::manifest::Manifest;
use crate::registry::ArtifactRegistry;
use tempfile::TempDir;

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

#[test]
fn test_apply_does_not_truncate_in_place_manifest() {
    let dir = TempDir::new().expect("temp dir");
    let manifests = dir.path().join("manifests");
    std::fs::create_dir_all(&manifests).expect("mkdir manifests");
    let manifest_json = sample_manifest_json("sample-v1", "sample_model", "safetensors");
    let manifest_path = manifests.join("sample-v1.json");
    std::fs::write(&manifest_path, &manifest_json).expect("write source manifest");

    let recipe_path = dir.path().join("recipe.json");
    let output_dir = dir.path().to_string_lossy();
    std::fs::write(
        &recipe_path,
        format!(
            r#"{{
          "recipe_id": "in-place-register",
          "type": "register",
          "inputs": {{ "source_manifest": "manifests/sample-v1.json" }},
          "outputs": {{ "output_dir": {output_dir} }}
        }}"#,
            output_dir = serde_json::to_string(&output_dir).unwrap()
        ),
    )
    .expect("write recipe");

    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    recipe
        .apply(Some(&dir.path().join("registry.json")))
        .expect("in-place register must not destroy the source manifest");

    let after = std::fs::read_to_string(&manifest_path).expect("read manifest after apply");
    assert_eq!(
        after, manifest_json,
        "copying a manifest onto itself must not truncate it"
    );
    Manifest::from_json(&after)
        .expect("parse")
        .validate()
        .expect("in-place apply must leave a valid manifest");
}

#[test]
fn test_failed_emit_does_not_write_registry() {
    let dir = TempDir::new().expect("temp dir");
    std::fs::write(
        dir.path().join("manifest.json"),
        sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    )
    .expect("write manifest");
    let blocked = dir.path().join("blocked");
    std::fs::write(&blocked, "not a directory").expect("block output_dir");
    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(
        &recipe_path,
        format!(
            r#"{{
          "recipe_id": "emit-fails",
          "type": "register",
          "inputs": {{ "source_manifest": "manifest.json" }},
          "outputs": {{ "output_dir": {} }}
        }}"#,
            serde_json::to_string(&blocked.to_string_lossy()).unwrap()
        ),
    )
    .expect("write recipe");

    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    let registry_path = dir.path().join("registry.json");
    recipe
        .apply(Some(&registry_path))
        .expect_err("emit must fail when output_dir is a file");
    assert!(
        !registry_path.exists(),
        "registry must not be published when artifact emission fails"
    );
}

#[test]
fn test_handoff_status_follows_combine_target() {
    let dir = TempDir::new().expect("temp dir");
    std::fs::write(
        dir.path().join("manifest.json"),
        sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    )
    .expect("write manifest");
    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(
        &recipe_path,
        r#"{
          "recipe_id": "placeholder-handoff",
          "type": "register",
          "inputs": { "source_manifest": "manifest.json" },
          "handoff": {
            "combine_for_ai": { "enabled": false, "status": "placeholder" }
          }
        }"#,
    )
    .expect("write recipe");

    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    let registry_path = dir.path().join("registry.json");
    recipe.apply(Some(&registry_path)).expect("apply");

    let handoff: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("handoff").join("sample-v1.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(handoff["benchmark_linkage"]["status"], "placeholder");
}

#[test]
fn test_unsafe_manifest_id_is_rejected_on_apply() {
    let (dir, recipe) = recipe_in_temp_dir(
        r#"{
          "recipe_id": "unsafe-id",
          "type": "register",
          "inputs": { "source_manifest": "manifest.json" }
        }"#,
        &sample_manifest_json("../escape", "sample_model", "safetensors"),
    );
    let err = recipe
        .apply(Some(&dir.path().join("registry.json")))
        .expect_err("path-like manifest_id must not be used as a filename");
    assert!(err.contains("filename-safe"), "{err}");
}

#[test]
fn test_explicit_null_is_rejected() {
    let (_dir, recipe) = recipe_in_temp_dir(
        r#"{
          "recipe_id": "null-format",
          "type": "register",
          "inputs": { "source_manifest": "manifest.json", "source_format": null }
        }"#,
        &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    );
    let err = recipe
        .validate()
        .expect_err("explicit null must not be treated as omitted");
    assert!(err.contains("must not be null"), "{err}");
}
