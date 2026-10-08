use crate::pack::*;
use magere_grok_process::types::GOZ1_VERSION;
use magere_grok_process::weight_pack::parse_pack;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

pub(super) const FIXTURE_DISSECT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../configs/recipes/fixtures/grok-mini-dissect.json"
);
pub(super) const SOURCE_MANIFEST: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../manifests/examples/grok-1-future-plan.json"
);
pub(super) const REDPAJAMA_MANIFEST: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../manifests/examples/redpajama-incite-7b-chat.json"
);
pub(super) const EXAMPLE_RECIPE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../configs/recipes/ternary-pack-example.json"
);
pub(super) const REPO_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// The fixture lists 6 ternary_candidates: 3 stay ternary, 2 match the `norm` fp16 rule
/// and 1 matches the `router` preserve rule (both encode as TENSOR_F16 on disk).
pub(super) const FIXTURE_TENSORS: u32 = 6;
pub(super) const FIXTURE_TERNARY: u32 = 3;
pub(super) const FIXTURE_F16: u32 = 3;

pub(super) struct Harness {
    pub(super) dir: TempDir,
    pub(super) recipe_path: PathBuf,
    pub(super) output_dir: PathBuf,
    pub(super) registry_path: PathBuf,
}

pub(super) fn write_recipe(dir: &Path, name: &str, recipe: &serde_json::Value) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, serde_json::to_string_pretty(recipe).unwrap()).unwrap();
    path
}

pub(super) fn harness_from(dir: TempDir, recipe: &serde_json::Value) -> Harness {
    let output_dir = dir.path().join("packs");
    let registry_path = dir.path().join("registry.json");
    let recipe_path = write_recipe(dir.path(), "recipe.json", recipe);
    Harness {
        dir,
        recipe_path,
        output_dir,
        registry_path,
    }
}

pub(super) fn base_recipe(output_dir: &Path) -> serde_json::Value {
    json!({
        "recipe_id": "test-ternary-pack",
        "type": "ternary_pack",
        "description": "unit-test recipe",
        "inputs": { "source_manifest": SOURCE_MANIFEST },
        "outputs": {
            "generated_format": "goz1",
            "manifest_id": "test-pack-v1",
            "output_dir": output_dir.display().to_string()
        },
        "pack": {
            "dissect_manifest": FIXTURE_DISSECT,
            "input_dir": "/models/grok-1",
            "input_format": "safetensors",
            "gif_threshold": 0.05,
            "use_embedded_baseline": false
        }
    })
}

pub(super) fn shipped_example_paths() -> (std::path::PathBuf, std::path::PathBuf, String, String) {
    let recipe = load_pack_recipe(Path::new(EXAMPLE_RECIPE)).expect("example recipe parses");
    let pack = recipe.pack.as_ref().expect("example carries a pack block");
    let root = Path::new(REPO_ROOT);
    let source_manifest = root.join(
        recipe
            .inputs
            .as_ref()
            .unwrap()
            .source_manifest
            .as_ref()
            .unwrap(),
    );
    let dissect = root.join(&pack.dissect_manifest);
    let source_family = Manifest::from_file(&source_manifest)
        .expect("source manifest parses")
        .model
        .family;
    let dissect_family = serde_json::from_str::<serde_json::Value>(
        &fs::read_to_string(&dissect).expect("dissect fixture is readable"),
    )
    .expect("dissect fixture is JSON")["model"]["family"]
        .as_str()
        .expect("dissect fixture names a model family")
        .to_string();
    (source_manifest, dissect, source_family, dissect_family)
}

pub(super) fn default_harness() -> Harness {
    let dir = TempDir::new().unwrap();
    let recipe = base_recipe(&dir.path().join("packs"));
    harness_from(dir, &recipe)
}

pub(super) fn assert_fixture_counts(outcome: &PackOutcome, bytes: &[u8]) {
    let (header, entries) = parse_pack(bytes).expect("parse_pack");
    assert_eq!(header.version, GOZ1_VERSION);
    assert_eq!(header.tensor_count, FIXTURE_TENSORS);
    assert_eq!(entries.len() as u32, FIXTURE_TENSORS);
    assert_eq!(outcome.tensor_count, FIXTURE_TENSORS);
    assert_eq!(outcome.ternary_count, FIXTURE_TERNARY);
    assert_eq!(outcome.f16_count, FIXTURE_F16);
    assert_eq!(outcome.size_bytes, bytes.len() as u64);
}

pub(super) fn assert_skeleton_manifest(outcome: &PackOutcome) {
    assert!(outcome.manifest.validate().is_ok());
    let generated = outcome.manifest.generated_artifact.as_ref().unwrap();
    assert_eq!(generated.format, "goz1");
    assert_eq!(generated.status.as_deref(), Some("planned"));
    assert_eq!(generated.version, Some(1));
    assert_eq!(
        generated.checksum.as_ref().unwrap().sha256.as_deref(),
        Some(outcome.sha256.as_str())
    );
    let summary = generated.tensor_summary.as_ref().unwrap();
    assert_eq!(summary.tensor_count, Some(FIXTURE_TENSORS));
    assert_eq!(summary.f16_count, Some(FIXTURE_F16));
    assert_eq!(summary.ternary_count, Some(FIXTURE_TERNARY));
    let lineage = generated.source_lineage.as_ref().unwrap();
    assert_eq!(
        lineage.manifest_id.as_deref(),
        Some("grok-1-future-plan-v1")
    );
    assert!(
        outcome
            .manifest
            .metadata
            .description
            .as_deref()
            .unwrap()
            .contains("SKELETON PACK")
    );
}
