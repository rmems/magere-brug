use super::io::verify_written_pack;
use super::recipe::input_format_for;
use super::*;
use crate::checksum;
use crate::registry::ArtifactRegistry;
use magere_grok_process::types::{GOZ1_VERSION, InputFormat};
use magere_grok_process::weight_pack::parse_pack;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

const FIXTURE_DISSECT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../configs/recipes/fixtures/grok-mini-dissect.json"
);
const SOURCE_MANIFEST: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../manifests/examples/grok-1-future-plan.json"
);
const REDPAJAMA_MANIFEST: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../manifests/examples/redpajama-incite-7b-chat.json"
);
const EXAMPLE_RECIPE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../configs/recipes/ternary-pack-example.json"
);
const REPO_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// The fixture lists 6 ternary_candidates: 3 stay ternary, 2 match the `norm` fp16 rule
/// and 1 matches the `router` preserve rule (both encode as TENSOR_F16 on disk).
const FIXTURE_TENSORS: u32 = 6;
const FIXTURE_TERNARY: u32 = 3;
const FIXTURE_F16: u32 = 3;

struct Harness {
    dir: TempDir,
    recipe_path: PathBuf,
    output_dir: PathBuf,
    registry_path: PathBuf,
}

fn write_recipe(dir: &Path, name: &str, recipe: &serde_json::Value) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, serde_json::to_string_pretty(recipe).unwrap()).unwrap();
    path
}

fn harness_from(dir: TempDir, recipe: &serde_json::Value) -> Harness {
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

fn base_recipe(output_dir: &Path) -> serde_json::Value {
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

fn default_harness() -> Harness {
    let dir = TempDir::new().unwrap();
    let recipe = base_recipe(&dir.path().join("packs"));
    harness_from(dir, &recipe)
}

fn assert_fixture_counts(outcome: &PackOutcome, bytes: &[u8]) {
    let (header, entries) = parse_pack(bytes).expect("parse_pack");
    assert_eq!(header.version, GOZ1_VERSION);
    assert_eq!(header.tensor_count, FIXTURE_TENSORS);
    assert_eq!(entries.len() as u32, FIXTURE_TENSORS);
    assert_eq!(outcome.tensor_count, FIXTURE_TENSORS);
    assert_eq!(outcome.ternary_count, FIXTURE_TERNARY);
    assert_eq!(outcome.f16_count, FIXTURE_F16);
    assert_eq!(outcome.size_bytes, bytes.len() as u64);
}

fn assert_skeleton_manifest(outcome: &PackOutcome) {
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
    let pack = recipe.pack.as_ref().expect("example carries a pack block");
    assert_eq!(pack.input_format, Some(InputFormat::NpyDir));

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
    assert!(source_manifest.exists(), "{}", source_manifest.display());
    assert!(dissect.exists(), "{}", dissect.display());

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
    assert_eq!(
        source_family, dissect_family,
        "shipped example pairs a '{source_family}' source manifest with a \
         '{dissect_family}' dissect fixture"
    );

    let dir = TempDir::new().unwrap();
    let mut as_absolute: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(EXAMPLE_RECIPE).unwrap()).unwrap();
    as_absolute["inputs"]["source_manifest"] = json!(source_manifest.display().to_string());
    as_absolute["pack"]["dissect_manifest"] = json!(dissect.display().to_string());
    let recipe_path = write_recipe(dir.path(), "example.json", &as_absolute);
    let registry_path = dir.path().join("registry.json");

    let outcome = run_pack_recipe(&recipe_path, Some(&registry_path), Some(dir.path()))
        .expect("shipped example recipe must run");
    assert!(fs::read(&outcome.pack_path).unwrap().starts_with(b"GOZ1"));
    assert_eq!(outcome.tensor_count, FIXTURE_TENSORS);
    assert_eq!(
        outcome.pack_path,
        dir.path().join("grok-1-goz1-pack-example-v1.goz1")
    );
    assert_eq!(
        outcome
            .manifest
            .generated_artifact
            .as_ref()
            .unwrap()
            .status
            .as_deref(),
        Some("planned")
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

fn valid_pack_bytes() -> Vec<u8> {
    let h = default_harness();
    let outcome = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap();
    fs::read(&outcome.pack_path).unwrap()
}

#[test]
fn verify_rejects_a_tail_truncated_pack() {
    let bytes = valid_pack_bytes();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("truncated.goz1");
    let truncated = &bytes[..bytes.len() - (FIXTURE_TENSORS as usize * 4)];
    fs::write(&path, truncated).unwrap();
    let (header, entries) = parse_pack(truncated).expect("truncated pack still parses");
    assert_eq!(header.tensor_count, FIXTURE_TENSORS);
    assert_eq!(entries.len() as u32, FIXTURE_TENSORS);
    let err = verify_written_pack(&path, &bytes).unwrap_err();
    assert!(err.contains("truncated"), "{}", err);
}

#[test]
fn verify_rejects_a_corrupted_byte() {
    let bytes = valid_pack_bytes();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("corrupt.goz1");
    let mut corrupt = bytes.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0xff;
    fs::write(&path, &corrupt).unwrap();
    let err = verify_written_pack(&path, &bytes).unwrap_err();
    assert!(err.contains("does not match"), "{}", err);
    assert!(err.contains(&last.to_string()), "{}", err);
}

#[test]
fn verify_rejects_a_pack_without_the_goz1_magic() {
    let bytes = valid_pack_bytes();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("nomagic.goz1");
    let mut clobbered = bytes.clone();
    clobbered[..4].copy_from_slice(b"NOPE");
    fs::write(&path, &clobbered).unwrap();
    let err = verify_written_pack(&path, &clobbered).unwrap_err();
    assert!(err.contains("GOZ1 magic"), "{}", err);
}

#[test]
fn verify_rejects_a_pack_with_the_wrong_version() {
    let bytes = valid_pack_bytes();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("badversion.goz1");
    let mut clobbered = bytes.clone();
    clobbered[4..8].copy_from_slice(&99u32.to_le_bytes());
    fs::write(&path, &clobbered).unwrap();
    let err = verify_written_pack(&path, &clobbered).unwrap_err();
    assert!(err.contains("did not round-trip"), "{}", err);
}

#[test]
fn verify_rejects_a_pack_with_an_unknown_dtype() {
    let bytes = valid_pack_bytes();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("baddtype.goz1");
    let mut clobbered = bytes.clone();
    let table_offset = u64::from_le_bytes(clobbered[12..20].try_into().unwrap()) as usize;
    let name_len = u16::from_le_bytes(
        clobbered[table_offset..table_offset + 2]
            .try_into()
            .unwrap(),
    ) as usize;
    clobbered[table_offset + 2 + name_len] = 0x7f;
    fs::write(&path, &clobbered).unwrap();
    let err = verify_written_pack(&path, &clobbered).unwrap_err();
    assert!(err.contains("unknown dtype"), "{}", err);
}

#[test]
fn verify_accepts_the_pack_the_command_actually_wrote() {
    let bytes = valid_pack_bytes();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("good.goz1");
    fs::write(&path, &bytes).unwrap();
    let stats = verify_written_pack(&path, &bytes).expect("valid pack verifies");
    assert_eq!(stats.tensor_count, FIXTURE_TENSORS);
    assert_eq!(stats.ternary_count, FIXTURE_TERNARY);
    assert_eq!(stats.f16_count, FIXTURE_F16);
    assert_eq!(stats.size_bytes, bytes.len() as u64);
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
fn a_mixed_pack_and_saaq_recipe_still_loads() {
    let dir = TempDir::new().unwrap();
    let mut recipe = base_recipe(&dir.path().join("packs"));
    recipe["saaq"] = json!({ "snn_steps": 4 });
    let h = harness_from(dir, &recipe);
    let parsed = load_pack_recipe(&h.recipe_path).expect("mixed recipe loads");
    assert!(parsed.saaq.is_some());
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
fn pack_goz1_command_renders_a_report() {
    let h = default_harness();
    let report = pack_goz1_command(&h.recipe_path, Some(&h.registry_path), None).unwrap();
    assert!(report.contains("GOZ1 pack written"));
    assert!(report.contains("generated_artifact.status: planned"));
    assert!(report.contains("SKELETON PACK"));
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
