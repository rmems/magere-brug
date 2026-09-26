use super::test_support::{recipe_in_temp_dir, sample_manifest_json, write_manifest_with_artifact};
use super::*;
use crate::manifest::Manifest;
use tempfile::TempDir;

#[test]
fn test_apply_rejects_symlinked_output_base() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("temp dir");
    write_manifest_with_artifact(
        &dir.path().join("manifest.json"),
        &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    );
    let outside = TempDir::new().expect("outside dir");
    symlink(outside.path(), dir.path().join("linked-out")).expect("symlink output base");

    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(
        &recipe_path,
        format!(
            r#"{{
          "recipe_id": "symlink-output-base",
          "type": "register",
          "inputs": {{ "source_manifest": "manifest.json" }},
          "outputs": {{ "output_dir": {} }}
        }}"#,
            serde_json::to_string(&dir.path().join("linked-out").to_string_lossy()).unwrap()
        ),
    )
    .expect("write recipe");

    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    let err = recipe
        .apply(Some(&dir.path().join("registry.json")))
        .expect_err("symlinked output base must not be followed");
    assert!(err.contains("refusing symlink"), "{err}");
    assert!(
        !outside.path().join("manifests").exists() && !outside.path().join("handoff").exists(),
        "apply must not write through a symlinked output base"
    );
}

#[test]
fn test_apply_rejects_symlinked_handoff_destination() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("temp dir");
    write_manifest_with_artifact(
        &dir.path().join("manifest.json"),
        &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    );
    std::fs::create_dir_all(dir.path().join("handoff")).expect("mkdir handoff");
    let outside = dir.path().join("outside.json");
    std::fs::write(&outside, b"secret").expect("write outside target");
    symlink(&outside, dir.path().join("handoff").join("sample-v1.json"))
        .expect("symlink handoff leaf");

    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(
        &recipe_path,
        r#"{
          "recipe_id": "symlink-handoff-leaf",
          "type": "register",
          "inputs": { "source_manifest": "manifest.json" }
        }"#,
    )
    .expect("write recipe");

    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    let err = recipe
        .apply(Some(&dir.path().join("registry.json")))
        .expect_err("symlinked handoff leaf must not be overwritten");
    assert!(err.contains("refusing symlink"), "{err}");
    assert_eq!(
        std::fs::read_to_string(&outside).expect("outside file unchanged"),
        "secret"
    );
}

#[test]
fn test_apply_rejects_dangling_symlink_at_handoff_destination() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("temp dir");
    write_manifest_with_artifact(
        &dir.path().join("manifest.json"),
        &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    );
    std::fs::create_dir_all(dir.path().join("handoff")).expect("mkdir handoff");
    symlink(
        dir.path().join("missing-target.json"),
        dir.path().join("handoff").join("sample-v1.json"),
    )
    .expect("dangling symlink handoff leaf");

    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(
        &recipe_path,
        r#"{
          "recipe_id": "dangling-symlink-handoff-leaf",
          "type": "register",
          "inputs": { "source_manifest": "manifest.json" }
        }"#,
    )
    .expect("write recipe");

    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    let err = recipe
        .apply(Some(&dir.path().join("registry.json")))
        .expect_err("dangling symlink at handoff leaf must not be followed");
    assert!(err.contains("refusing symlink"), "{err}");
    assert!(
        !dir.path().join("missing-target.json").exists(),
        "apply must not create the dangling symlink target"
    );
}

#[test]
fn test_apply_does_not_truncate_in_place_manifest() {
    let dir = TempDir::new().expect("temp dir");
    let manifests = dir.path().join("manifests");
    std::fs::create_dir_all(&manifests).expect("mkdir manifests");
    let manifest_json = sample_manifest_json("sample-v1", "sample_model", "safetensors");
    let manifest_path = manifests.join("sample-v1.json");
    write_manifest_with_artifact(&manifest_path, &manifest_json);

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
    write_manifest_with_artifact(
        &dir.path().join("manifest.json"),
        &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    );
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
    write_manifest_with_artifact(
        &dir.path().join("manifest.json"),
        &sample_manifest_json("sample-v1", "sample_model", "safetensors"),
    );
    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(
        &recipe_path,
        r#"{
          "recipe_id": "placeholder-handoff",
          "type": "register",
          "inputs": { "source_manifest": "manifest.json" },
          "handoff": {
            "combine_for_ai": { "enabled": false, "status": "ready" }
          }
        }"#,
    )
    .expect("write recipe");

    let recipe = Recipe::from_file(&recipe_path).expect("load recipe");
    assert!(
        recipe
            .summary()
            .contains("status=placeholder, enabled=false"),
        "{}",
        recipe.summary()
    );
    let registry_path = dir.path().join("registry.json");
    recipe.apply(Some(&registry_path)).expect("apply");

    let handoff: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("handoff").join("sample-v1.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(handoff["benchmark_linkage"]["status"], "placeholder");
}

#[test]
fn test_missing_local_source_artifact_is_rejected_before_publication() {
    let dir = TempDir::new().expect("temp dir");
    std::fs::write(
        dir.path().join("manifest.json"),
        sample_manifest_json("missing-v1", "missing_model", "gguf"),
    )
    .expect("write manifest only");
    let recipe_path = dir.path().join("recipe.json");
    std::fs::write(
        &recipe_path,
        r#"{
          "recipe_id": "missing-local-artifact",
          "type": "register",
          "inputs": { "source_manifest": "manifest.json" }
        }"#,
    )
    .expect("write recipe");
    let recipe = Recipe::from_file(recipe_path).expect("load recipe");
    let registry_path = dir.path().join("registry.json");

    let error = recipe
        .apply(Some(&registry_path))
        .expect_err("missing local source must be rejected");
    assert!(error.contains("not an existing file"), "{error}");
    assert!(!registry_path.exists());
    assert!(!dir.path().join("manifests").exists());
    assert!(!dir.path().join("handoff").exists());
}

#[test]
fn test_non_file_source_formats_do_not_require_local_artifacts() {
    for source_format in ["hf_repo", "local_dir"] {
        let (dir, recipe) = recipe_in_temp_dir(
            r#"{
              "recipe_id": "source-only-register",
              "type": "register",
              "inputs": { "source_manifest": "manifest.json" }
            }"#,
            &sample_manifest_json("source-only-v1", "source_only", source_format),
        );
        recipe
            .apply(Some(&dir.path().join("registry.json")))
            .unwrap_or_else(|error| panic!("{source_format} should remain source-only: {error}"));
    }
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
