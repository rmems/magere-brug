//! Generated manifest construction and CLI report rendering.

use super::SKELETON_NOTICE;
use super::io::PackStats;
use super::recipe::PackRecipe;
use crate::manifest::{
    BackendStatus, Checksum, GeneratedArtifact, Manifest, Metadata, Quantization, SourceLineage,
    TensorSummary,
};
use magere_grok_process::types::GOZ1_VERSION;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// `generated_artifact.status` recorded for skeleton packs.
const SKELETON_STATUS: &str = "planned";

/// The dissect manifest that determined the pack's tensor table, kept for lineage.
pub(super) struct DissectInput {
    pub path: PathBuf,
    pub sha256: String,
}

/// Inputs for [`build_generated_manifest`].
pub(super) struct GeneratedManifestSpec<'a> {
    pub recipe: &'a PackRecipe,
    pub source: &'a Manifest,
    pub manifest_id: String,
    pub pack_path: &'a Path,
    pub sha256: &'a str,
    pub stats: PackStats,
    pub dissect_input: &'a DissectInput,
    pub resolved_input_dir: &'a str,
}

/// Build the model manifest that records the pack as a `generated_artifact`.
pub(super) fn build_generated_manifest(spec: GeneratedManifestSpec<'_>) -> Manifest {
    let GeneratedManifestSpec {
        recipe,
        source,
        manifest_id,
        pack_path,
        sha256,
        stats,
        dissect_input,
        resolved_input_dir,
    } = spec;
    let now = chrono::Utc::now().to_rfc3339();
    let mut model = source.model.clone();
    model.slug = format!("{}_goz1", source.model.slug);
    Manifest {
        metadata: Metadata {
            schema_version: source.metadata.schema_version,
            created_at: now.clone(),
            manifest_id,
            description: Some(manifest_description(recipe)),
        },
        model,
        source_artifact: source.source_artifact.clone(),
        generated_artifact: Some(generated_artifact(
            pack_path,
            sha256,
            stats,
            now,
            lineage_for(source, dissect_input, resolved_input_dir),
        )),
        quantization: Some(Quantization {
            method: Some("ternary".to_string()),
            bits: Some(2),
            group_size: None,
            calibration_dataset: None,
            calibration_config_path: None,
        }),
        backend_compatibility: Some(goz1_backends()),
        saaq_experiment: None,
        benchmark_linkage: None,
    }
}

fn manifest_description(recipe: &PackRecipe) -> String {
    let recipe_note = match &recipe.description {
        Some(description) => format!(" Recipe description: {}", description),
        None => String::new(),
    };
    format!(
        "GOZ1 pack emitted by `magere pack-goz1` from recipe '{}'.{} {}",
        recipe.recipe_id, recipe_note, SKELETON_NOTICE
    )
}

fn generated_artifact(
    pack_path: &Path,
    sha256: &str,
    stats: PackStats,
    now: String,
    source_lineage: SourceLineage,
) -> GeneratedArtifact {
    GeneratedArtifact {
        format: "goz1".to_string(),
        path: Some(pack_path.display().to_string()),
        status: Some(SKELETON_STATUS.to_string()),
        version: Some(GOZ1_VERSION),
        source_url: None,
        checksum: Some(Checksum {
            sha256: Some(sha256.to_string()),
            md5: None,
        }),
        dtype_summary: None,
        size_bytes: Some(stats.size_bytes),
        shard_info: None,
        timestamp: Some(now),
        source_lineage: Some(source_lineage),
        tensor_summary: Some(TensorSummary {
            tensor_count: Some(stats.tensor_count),
            f16_count: Some(stats.f16_count),
            ternary_count: Some(stats.ternary_count),
        }),
    }
}

fn lineage_for(
    source: &Manifest,
    dissect_input: &DissectInput,
    resolved_input_dir: &str,
) -> SourceLineage {
    let checksum = if resolved_input_dir == source.source_artifact.path {
        source.source_artifact.checksum.clone()
    } else {
        None
    };
    SourceLineage {
        manifest_id: Some(source.metadata.manifest_id.clone()),
        path: Some(resolved_input_dir.to_string()),
        checksum,
        dissect_manifest_path: Some(dissect_input.path.display().to_string()),
        dissect_manifest_checksum: Some(Checksum {
            sha256: Some(dissect_input.sha256.clone()),
            md5: None,
        }),
    }
}

fn goz1_backends() -> HashMap<String, BackendStatus> {
    let mut backends = HashMap::new();
    backends.insert(
        "goz1".to_string(),
        BackendStatus {
            supported: Some(true),
            status: Some("planned".to_string()),
            kernel_types: None,
        },
    );
    backends
}

/// Render the human-readable CLI report, skeleton caveat included.
pub(super) fn format_outcome(outcome: &super::PackOutcome) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "✓ GOZ1 pack written from recipe '{}': {}\n",
        outcome.recipe_id,
        outcome.pack_path.display()
    ));
    out.push_str(&format!(
        "  magic: GOZ1  version: {}  tensors: {} (ternary {}, f16/preserve {})  size: {} bytes\n",
        GOZ1_VERSION,
        outcome.tensor_count,
        outcome.ternary_count,
        outcome.f16_count,
        outcome.size_bytes
    ));
    out.push_str(&format!("  sha256: {}\n", outcome.sha256));
    out.push_str(&format!(
        "  manifest: {} (manifest_id: {}, slug: {}, generated_artifact.status: {})\n",
        outcome.manifest_path.display(),
        outcome.manifest.metadata.manifest_id,
        outcome.manifest.model.slug,
        outcome
            .manifest
            .generated_artifact
            .as_ref()
            .and_then(|g| g.status.as_deref())
            .unwrap_or("-")
    ));
    out.push_str(&format!(
        "  registry: {} ({})\n",
        outcome.registry_path.display(),
        if outcome.registry_replaced {
            "entry replaced"
        } else {
            "entry added"
        }
    ));
    if outcome.input_dir_missing {
        out.push_str(
            "  note: pack.input_dir does not exist on disk; the skeleton packer never reads it\n",
        );
    }
    out.push_str(&format!("! {}\n", SKELETON_NOTICE));
    out
}
