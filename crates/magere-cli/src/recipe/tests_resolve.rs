use super::resolve::{is_filename_safe_id, reference_escapes_upward};
use super::*;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repo root is reachable from CARGO_MANIFEST_DIR")
}

#[test]
fn reference_does_not_resolve_above_the_repo_root() {
    let lab = tempfile::tempdir().expect("tempdir");
    let outside = lab.path().join("manifests").join("examples");
    std::fs::create_dir_all(&outside).expect("mkdir outside");
    std::fs::copy(
        repo_root()
            .join("manifests")
            .join("examples")
            .join("olmoe-1b-7b-instruct.json"),
        outside.join("planted.json"),
    )
    .expect("plant manifest above the repo root");

    let recipes = lab.path().join("fakerepo").join("configs").join("recipes");
    std::fs::create_dir_all(&recipes).expect("mkdir fakerepo");
    std::fs::write(lab.path().join("fakerepo").join("Cargo.lock"), "").expect("repo marker");

    let recipe_path = recipes.join("escape.json");
    std::fs::write(
        &recipe_path,
        r#"{"recipe_id":"escape-test","type":"register",
            "inputs":{"source_manifest":"manifests/examples/planted.json"},
            "outputs":{"manifest_id":"olmoe-1b-7b-instruct-v1"}}"#,
    )
    .expect("write recipe");

    let recipe = Recipe::from_file(&recipe_path).expect("recipe loads");
    let err = recipe
        .validate()
        .expect_err("a reference above the repo root must not resolve");
    assert!(
        err.contains("could not be resolved"),
        "unexpected error: {err}"
    );
}

#[test]
fn reference_with_parent_dir_segments_is_rejected() {
    let recipe = Recipe::from_json(
        r#"{"recipe_id":"dotdot-test","type":"register",
            "inputs":{"source_manifest":"../../secret/outside.json"},
            "outputs":{"manifest_id":"whatever-v1"}}"#,
    )
    .expect("recipe parses");

    let err = recipe.validate().expect_err("'..' must be rejected");
    assert!(
        err.contains("must not contain '..'"),
        "unexpected error: {err}"
    );
}

/// Dots inside a filename are not path traversal.
#[test]
fn dots_inside_a_filename_are_not_treated_as_traversal() {
    assert!(!reference_escapes_upward(
        "manifests/examples/model..v2.json"
    ));
    assert!(!reference_escapes_upward("manifests/examples/m.json"));
    assert!(reference_escapes_upward("../m.json"));
    assert!(reference_escapes_upward("a/../../m.json"));
}

#[test]
fn validate_reports_the_file_each_reference_resolved_to() {
    let path = repo_root()
        .join("configs")
        .join("recipes")
        .join("register-gguf-example.json");
    let recipe = Recipe::from_file(&path).expect("recipe loads");
    recipe.validate().expect("example is valid");

    let resolved = recipe.resolved_references();
    assert_eq!(resolved.len(), 1, "expected one resolved input reference");
    assert_eq!(resolved[0].0, "inputs.source_manifest");
    assert!(
        resolved[0].1.ends_with("olmoe-1b-7b-instruct.json"),
        "unexpected resolution: {}",
        resolved[0].1.display()
    );
}

/// The bound must hold even when no ancestor carries a repository marker.
/// The earlier implementation only stopped *at* a marker, so a marker-less
/// tree still walked to `/` — and every other test here creates a marker,
/// so nothing covered it.
#[test]
fn reference_does_not_escape_a_tree_with_no_repo_marker() {
    let lab = tempfile::tempdir().expect("tempdir");
    let outside = lab.path().join("manifests").join("examples");
    std::fs::create_dir_all(&outside).expect("mkdir outside");
    std::fs::copy(
        repo_root()
            .join("manifests")
            .join("examples")
            .join("olmoe-1b-7b-instruct.json"),
        outside.join("planted.json"),
    )
    .expect("plant manifest above the recipe");

    // Deliberately NO .git and NO Cargo.lock anywhere in this tree.
    let recipes = lab.path().join("looserepo").join("configs").join("recipes");
    std::fs::create_dir_all(&recipes).expect("mkdir looserepo");

    let recipe_path = recipes.join("nomarker.json");
    std::fs::write(
        &recipe_path,
        r#"{"recipe_id":"nomarker-test","type":"register",
            "inputs":{"source_manifest":"manifests/examples/planted.json"},
            "outputs":{"manifest_id":"olmoe-1b-7b-instruct-v1"}}"#,
    )
    .expect("write recipe");

    let recipe = Recipe::from_file(&recipe_path).expect("recipe loads");
    let err = recipe
        .validate()
        .expect_err("a marker-less tree must not resolve references above the recipe");
    assert!(
        err.contains("could not be resolved"),
        "unexpected error: {err}"
    );
}

/// A reference sitting beside the recipe still resolves in a marker-less tree.
#[test]
fn marker_less_tree_still_resolves_a_sibling_reference() {
    let lab = tempfile::tempdir().expect("tempdir");
    let dir = lab.path().join("loose");
    std::fs::create_dir_all(&dir).expect("mkdir loose");
    std::fs::copy(
        repo_root()
            .join("manifests")
            .join("examples")
            .join("olmoe-1b-7b-instruct.json"),
        dir.join("beside.json"),
    )
    .expect("copy manifest beside the recipe");

    let recipe_path = dir.join("sibling.json");
    std::fs::write(
        &recipe_path,
        r#"{"recipe_id":"sibling-test","type":"register",
            "inputs":{"source_manifest":"beside.json"},
            "outputs":{"manifest_id":"olmoe-1b-7b-instruct-v1"}}"#,
    )
    .expect("write recipe");

    let recipe = Recipe::from_file(&recipe_path).expect("recipe loads");
    recipe
        .validate()
        .expect("a sibling reference must still resolve");
}

#[test]
fn relative_recipe_path_still_resolves_repo_root_refs() {
    let root = repo_root();
    let previous = std::env::current_dir().expect("cwd");
    std::env::set_current_dir(&root).expect("cd repo root");
    let result = {
        let recipe = Recipe::from_file("configs/recipes/gguf-export-placeholder-example.json")
            .expect("relative recipe path loads");
        recipe.validate()
    };
    std::env::set_current_dir(previous).expect("restore cwd");
    result.expect("relative recipe paths must still resolve repo-root manifests");
}

#[test]
fn filename_safe_ids_reject_path_components() {
    assert!(is_filename_safe_id("sample-v1"));
    assert!(is_filename_safe_id("model.v2"));
    assert!(!is_filename_safe_id("../escape"));
    assert!(!is_filename_safe_id("a/b"));
    assert!(!is_filename_safe_id("/tmp/x"));
    assert!(!is_filename_safe_id(""));
}

#[test]
fn absolute_manifest_reference_is_rejected() {
    let lab = tempfile::tempdir().expect("tempdir");
    let outside = lab.path().join("outside.json");
    std::fs::copy(
        repo_root()
            .join("manifests")
            .join("examples")
            .join("olmoe-1b-7b-instruct.json"),
        &outside,
    )
    .expect("plant absolute manifest");

    let dir = lab.path().join("repo");
    std::fs::create_dir_all(&dir).expect("mkdir repo");
    std::fs::write(dir.join("Cargo.lock"), "").expect("repo marker");
    let recipe_path = dir.join("abs.json");
    std::fs::write(
        &recipe_path,
        format!(
            r#"{{"recipe_id":"abs-test","type":"register",
            "inputs":{{"source_manifest":{}}},
            "outputs":{{"manifest_id":"olmoe-1b-7b-instruct-v1"}}}}"#,
            serde_json::to_string(&outside.to_string_lossy()).unwrap()
        ),
    )
    .expect("write recipe");

    let recipe = Recipe::from_file(&recipe_path).expect("recipe loads");
    let err = recipe
        .validate()
        .expect_err("absolute source_manifest must be rejected");
    assert!(err.contains("absolute"), "unexpected error: {err}");
}

#[test]
fn symlink_escape_is_rejected() {
    let lab = tempfile::tempdir().expect("tempdir");
    let outside = lab.path().join("planted.json");
    std::fs::copy(
        repo_root()
            .join("manifests")
            .join("examples")
            .join("olmoe-1b-7b-instruct.json"),
        &outside,
    )
    .expect("plant outside manifest");

    let dir = lab.path().join("repo");
    std::fs::create_dir_all(&dir).expect("mkdir repo");
    std::fs::write(dir.join("Cargo.lock"), "").expect("repo marker");
    std::os::unix::fs::symlink(&outside, dir.join("link.json")).expect("symlink");

    let recipe_path = dir.join("link-recipe.json");
    std::fs::write(
        &recipe_path,
        r#"{"recipe_id":"symlink-test","type":"register",
            "inputs":{"source_manifest":"link.json"},
            "outputs":{"manifest_id":"olmoe-1b-7b-instruct-v1"}}"#,
    )
    .expect("write recipe");

    let recipe = Recipe::from_file(&recipe_path).expect("recipe loads");
    let err = recipe
        .validate()
        .expect_err("a symlink that leaves the repository must not resolve");
    assert!(
        err.contains("could not be resolved") || err.contains("absolute"),
        "unexpected error: {err}"
    );
}
