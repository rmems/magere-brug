use super::*;
use crate::checksum::compute_string_sha256;
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

fn recipe_in_temp_dir(recipe_json: &str, manifest_json: &str) -> (TempDir, Recipe) {
    let dir = TempDir::new().expect("temp dir");
    std::fs::write(dir.path().join("manifest.json"), manifest_json).expect("write manifest");
    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(&recipe_path, recipe_json).expect("write recipe");
    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    (dir, recipe)
}

#[test]
fn test_apply_saaq_delegates_to_runner() {
    let dir = TempDir::new().expect("temp dir");
    std::fs::write(
        dir.path().join("manifest.json"),
        sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    )
    .expect("write manifest");
    let output_dir = dir.path().join("saaq-run");
    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(
        &recipe_path,
        format!(
            r#"{{
          "recipe_id": "saaq-delegates",
          "type": "saaq",
          "inputs": {{ "source_manifest": "manifest.json" }},
          "outputs": {{ "output_dir": {} }}
        }}"#,
            serde_json::to_string(&output_dir.to_string_lossy()).unwrap()
        ),
    )
    .expect("write recipe");
    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    let output = recipe.apply(None).expect("saaq apply should run");
    assert!(output.contains("SAAQ run"), "{output}");
    assert!(output_dir.join("latent_telemetry.csv").is_file());
    assert!(output_dir.join("run_manifest.json").is_file());
}

#[test]
fn test_apply_saaq_hashes_original_recipe_bytes() {
    let dir = TempDir::new().expect("temp dir");
    std::fs::write(
        dir.path().join("manifest.json"),
        sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    )
    .expect("write manifest");
    let output_dir = dir.path().join("saaq-run");
    let recipe_path = dir.path().join("recipe.json");
    // Pretty-printed, key-ordered differently from serde_json::to_string so a
    // reserialized digest cannot accidentally match the file on disk.
    let recipe_text = format!(
        "{{\n  \"type\": \"saaq\",\n  \"recipe_id\": \"saaq-digest\",\n  \"inputs\": {{\n    \"source_manifest\": \"manifest.json\"\n  }},\n  \"outputs\": {{\n    \"output_dir\": {}\n  }}\n}}\n",
        serde_json::to_string(&output_dir.to_string_lossy()).unwrap()
    );
    std::fs::write(&recipe_path, &recipe_text).expect("write recipe");

    Recipe::from_file(&recipe_path)
        .expect("load recipe")
        .apply(None)
        .expect("saaq apply should run");

    let run_manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(output_dir.join("run_manifest.json")).expect("read run manifest"),
    )
    .expect("parse run manifest");
    let recorded = run_manifest["recipe"]["sha256"]
        .as_str()
        .expect("recipe sha256");
    assert_eq!(
        recorded,
        compute_string_sha256(&recipe_text),
        "recipe apply must hash the loaded file bytes, not a reserialized Value"
    );

    let via_run_saaq = crate::saaq::SaaqRunConfig::load(&recipe_path, None)
        .expect("direct run-saaq load")
        .recipe_sha256;
    assert_eq!(
        recorded, via_run_saaq,
        "recipe apply and magere run-saaq must record the same recipe digest"
    );
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

#[test]
fn test_saaq_register_true_is_rejected() {
    let dir = TempDir::new().expect("temp dir");
    std::fs::write(
        dir.path().join("manifest.json"),
        sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    )
    .expect("write manifest");
    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(
        &recipe_path,
        format!(
            r#"{{
          "recipe_id": "saaq-register",
          "type": "saaq",
          "inputs": {{ "source_manifest": "manifest.json" }},
          "outputs": {{ "output_dir": {}, "register": true }}
        }}"#,
            serde_json::to_string(&dir.path().join("out").to_string_lossy()).unwrap()
        ),
    )
    .expect("write recipe");
    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    let err = recipe
        .validate()
        .expect_err("saaq register:true must not silently succeed");
    assert!(err.contains("do not register"), "{err}");
}

#[test]
fn test_saaq_calibration_is_rejected() {
    let dir = TempDir::new().expect("temp dir");
    std::fs::write(
        dir.path().join("manifest.json"),
        sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    )
    .expect("write manifest");
    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(
        &recipe_path,
        format!(
            r#"{{
          "recipe_id": "saaq-cal",
          "type": "saaq",
          "inputs": {{ "source_manifest": "manifest.json" }},
          "outputs": {{ "output_dir": {} }},
          "calibration": {{ "dataset": "wikitext-2" }}
        }}"#,
            serde_json::to_string(&dir.path().join("out").to_string_lossy()).unwrap()
        ),
    )
    .expect("write recipe");
    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    let err = recipe
        .validate()
        .expect_err("unconsumed SAAQ calibration must be rejected");
    assert!(err.contains("calibration"), "{err}");
}

#[test]
fn test_saaq_artifact_path_is_rejected() {
    let dir = TempDir::new().expect("temp dir");
    std::fs::write(
        dir.path().join("manifest.json"),
        sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    )
    .expect("write manifest");
    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(
        &recipe_path,
        format!(
            r#"{{
          "recipe_id": "saaq-artifact-path",
          "type": "saaq",
          "inputs": {{ "source_manifest": "manifest.json" }},
          "outputs": {{ "output_dir": {}, "artifact_path": "unused.bin" }}
        }}"#,
            serde_json::to_string(&dir.path().join("out").to_string_lossy()).unwrap()
        ),
    )
    .expect("write recipe");
    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    let error = recipe
        .validate()
        .expect_err("SAAQ must reject an output it does not produce");
    assert!(error.contains("artifact_path"), "{error}");
}

#[test]
fn test_saaq_knobs_are_checked_during_recipe_validate() {
    let dir = TempDir::new().expect("temp dir");
    std::fs::write(
        dir.path().join("manifest.json"),
        sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    )
    .expect("write manifest");
    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(
        &recipe_path,
        format!(
            r#"{{
          "recipe_id": "saaq-knobs",
          "type": "saaq",
          "inputs": {{ "source_manifest": "manifest.json" }},
          "outputs": {{ "output_dir": {} }},
          "saaq": {{ "num_experts": 2, "top_k": 3 }}
        }}"#,
            serde_json::to_string(&dir.path().join("out").to_string_lossy()).unwrap()
        ),
    )
    .expect("write recipe");
    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    let err = recipe
        .validate()
        .expect_err("SAAQ cross-field knobs must fail recipe validate");
    assert!(
        err.contains("top_k") || err.contains("num_experts"),
        "{err}"
    );
}

#[test]
fn test_run_saaq_enforces_source_format() {
    let dir = TempDir::new().expect("temp dir");
    std::fs::write(
        dir.path().join("manifest.json"),
        sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    )
    .expect("write manifest");
    let recipe_path = dir.path().join("recipe.json");
    let output_dir = dir.path().join("out");
    std::fs::write(
        &recipe_path,
        format!(
            r#"{{
          "recipe_id": "saaq-format",
          "type": "saaq",
          "inputs": {{ "source_manifest": "manifest.json", "source_format": "gguf" }},
          "outputs": {{ "output_dir": {} }}
        }}"#,
            serde_json::to_string(&output_dir.to_string_lossy()).unwrap()
        ),
    )
    .expect("write recipe");
    let err = crate::saaq::run_saaq_command(&recipe_path, None)
        .expect_err("direct run-saaq must honor source_format");
    assert!(err.contains("source_format"), "{err}");
}
