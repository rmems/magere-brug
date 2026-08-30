//! Recipe-driven SAAQ validation runner (post-GOZ1).
//!
//! Reads a `type: "saaq"` recipe (`schemas/recipe.schema.json`), replays a
//! deterministic telemetry stream through the CPU-only `magere-corinth-core`
//! validation pipeline and writes two run artifacts:
//!
//! * `<output_dir>/latent_telemetry.csv` — one `SnnLatentSnapshot` row per tick,
//!   in the exact column layout emitted by [`SnnLatentCsvExporter`].
//! * `<output_dir>/run_manifest.json` — the reproducibility record for the run
//!   (resolved input refs, every effective SAAQ parameter, tick count, CSV path
//!   + SHA256, crate versions, wall-clock `created_at`).
//!
//! Pipeline per tick:
//!
//! ```text
//! TelemetrySnapshot
//!   -> TelemetryFunnel::encode_snapshot   (ternary events -> input spikes -> GIF hidden layer)
//!   -> Projector::project                 (spike train + potentials -> dense embedding)
//!   -> ModelOutput                        (embedding -> deterministic expert weights)
//!   -> SnnLatentCalibrator::observe       (SAAQ latent snapshot)
//!   -> SnnLatentCsvExporter::write_row
//! ```
//!
//! This is a **validation** pass, not a fitting pass: the projector matrix and
//! the funnel weights are fixed constants of the crate, nothing in this module
//! mutates them, and no signal is propagated backwards. The only state that
//! evolves across ticks is the pipeline's own membrane / adaptation / baseline
//! state, which is exactly what the SAAQ columns are there to observe.
//!
//! # Determinism
//!
//! Given the same recipe, `latent_telemetry.csv` is byte-identical across runs
//! and machines-of-the-same-float-behaviour:
//!
//! * the `synthetic` telemetry source is a pure function of the tick index and
//!   the recipe-supplied start/delta/interval values — no RNG, no wall clock;
//! * `timestamp_ms` comes from `start_timestamp_ms + tick_interval_ms * tick`,
//!   never from the system clock;
//! * expert weights are a pure function of the embedding (see
//!   [`expert_weights_from_embedding`]);
//! * the only wall-clock value produced by the runner (`created_at`) lives in
//!   `run_manifest.json`, deliberately kept out of the CSV.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use magere_corinth_core::projector::Projector;
use magere_corinth_core::{
    FUNNEL_HIDDEN_NEURONS, ModelOutput, ProjectionMode, SaaqUpdateRule, SnnDualLatentCalibrator,
    SnnLatentCalibrator, SnnLatentCsvExporter, SnnLatentSnapshot, TelemetryFunnel,
    TelemetrySnapshot,
};
use serde::Deserialize;

use crate::checksum;

// ── Defaults ──────────────────────────────────────────────────────────────

/// SNN time-steps expanded per telemetry snapshot when the recipe is silent.
const DEFAULT_SNN_STEPS: usize = 20;
/// Per-channel `TelemetryEncoder` thresholds when the recipe is silent.
const DEFAULT_THRESHOLDS: [f32; 4] = [1.0, 5.0, 1.0, 5.0];
/// Expert count used to derive routing entropy when the recipe is silent.
const DEFAULT_NUM_EXPERTS: usize = 8;
/// Experts recorded in `selected_experts` when the recipe is silent.
const DEFAULT_TOP_K: usize = 1;
/// Synthetic telemetry ticks when the recipe is silent.
const DEFAULT_TICKS: usize = 16;
/// Milliseconds between synthetic telemetry ticks when the recipe is silent.
const DEFAULT_TICK_INTERVAL_MS: u64 = 250;

const LATENT_CSV_FILE: &str = "latent_telemetry.csv";
const RUN_MANIFEST_FILE: &str = "run_manifest.json";
const RUN_MANIFEST_SCHEMA: &str = "magere-brug/saaq-run/1";

/// Human-readable description of the `expert_weights` derivation, copied into
/// every run manifest so a reader never has to guess how routing entropy was
/// produced.
///
/// This string is the only disclosure some readers ever see, so it states the
/// surrogate up front: a run manifest that also names a real MoE source
/// manifest and a GOZ1 ref would otherwise read as if `routing_entropy` came
/// from the model's own router.
const EXPERT_WEIGHT_SCHEME: &str = "PLACEHOLDER SURROGATE - not the model's real MoE router. \
     Softmax over the per-slice means of num_experts contiguous equal-width slices of the \
     projector embedding. Derived only from telemetry-driven SNN activity; no model weights \
     are read. Over a typical run its entropy stays within ~1% of the uniform maximum, so \
     routing_entropy is near-constant and must not be treated as a live routing signal.";

/// Accepted `outputs.generated_format` values, mirroring the enum in
/// `schemas/recipe.schema.json`.
const GENERATED_FORMATS: [&str; 4] = ["goz1", "gguf", "ternary", "binary"];

/// Columns a `csv` telemetry source must provide, in any order.
const TELEMETRY_CSV_COLUMNS: [&str; 5] = [
    "timestamp_ms",
    "gpu_temp_c",
    "gpu_power_w",
    "cpu_tctl_c",
    "cpu_package_power_w",
];

// ── Recipe deserialization ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRecipe {
    #[serde(default)]
    recipe_id: Option<String>,
    #[serde(rename = "type", default)]
    recipe_type: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    inputs: Option<RawInputs>,
    #[serde(default)]
    outputs: Option<RawOutputs>,
    #[serde(default)]
    saaq: Option<RawSaaq>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInputs {
    #[serde(default)]
    source_manifest: Option<String>,
    #[serde(default)]
    goz1_ref: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOutputs {
    #[serde(default)]
    generated_format: Option<String>,
    #[serde(default)]
    manifest_id: Option<String>,
    #[serde(default)]
    output_dir: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSaaq {
    #[serde(default)]
    projection_mode: Option<String>,
    #[serde(default)]
    snn_steps: Option<i64>,
    #[serde(default)]
    thresholds: Option<Vec<f32>>,
    #[serde(default)]
    update_rule: Option<String>,
    #[serde(default)]
    dual_rule: Option<bool>,
    #[serde(default)]
    num_experts: Option<i64>,
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    telemetry: Option<RawTelemetry>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTelemetry {
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    ticks: Option<i64>,
    #[serde(default)]
    tick_interval_ms: Option<i64>,
    #[serde(default)]
    start_timestamp_ms: Option<i64>,
    #[serde(default)]
    start: Option<Channels>,
    #[serde(default)]
    delta: Option<Channels>,
    #[serde(default)]
    path: Option<String>,
}

/// The four telemetry channels the `TelemetryEncoder` consumes.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Channels {
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub cpu_tctl_c: f32,
    pub cpu_package_power_w: f32,
}

// ── Effective (validated) run configuration ───────────────────────────────

/// A recipe input reference together with the file it resolved to on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRef {
    /// The reference exactly as written in the recipe.
    pub reference: String,
    /// The existing file the reference resolved to.
    pub resolved_path: PathBuf,
    /// SHA256 of the file at the same point validation resolved it.
    pub sha256: String,
}

/// Where the runner gets its `TelemetrySnapshot` stream from.
#[derive(Debug, Clone, PartialEq)]
pub enum TelemetrySource {
    /// Snapshots synthesised as a pure function of the tick index.
    Synthetic(SyntheticTelemetry),
    /// Snapshots replayed from a telemetry CSV on disk.
    ///
    /// The rows are parsed once, during validation, and carried here: re-reading
    /// the file at execution time would both duplicate the parse and leave a
    /// TOCTOU window in which the validated bytes and the executed bytes differ.
    Csv {
        path: PathBuf,
        snapshots: Vec<TelemetrySnapshot>,
        /// SHA256 of the exact UTF-8 byte buffer parsed into `snapshots`.
        ///
        /// Keeping the digest beside the retained rows prevents the run
        /// manifest from attributing them to a later version of a telemetry
        /// file that changed after validation.
        sha256: String,
    },
}

/// Parameters of the deterministic synthetic telemetry ramp.
///
/// Channel `c` at tick `i` is `start.c + delta.c * i`, and the tick timestamp
/// is `start_timestamp_ms + tick_interval_ms * i`. Because the encoder only
/// re-baselines on a threshold crossing, a sub-threshold `delta` produces a
/// periodic (rather than constant) spike pattern, which is usually what you
/// want out of a validation ramp.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticTelemetry {
    pub ticks: usize,
    pub tick_interval_ms: u64,
    pub start_timestamp_ms: u64,
    pub start: Channels,
    pub delta: Channels,
}

/// Every effective parameter of a SAAQ run, after defaults and validation.
#[derive(Debug, Clone, PartialEq)]
pub struct SaaqRunConfig {
    pub recipe_id: String,
    pub recipe_path: PathBuf,
    pub description: Option<String>,
    pub source_manifest: Option<ResolvedRef>,
    pub goz1_ref: Option<ResolvedRef>,
    pub generated_format: Option<String>,
    pub manifest_id: Option<String>,
    pub output_dir: PathBuf,
    pub projection_mode: ProjectionMode,
    pub snn_steps: usize,
    pub thresholds: [f32; 4],
    pub update_rule: SaaqUpdateRule,
    pub dual_rule: bool,
    pub num_experts: usize,
    /// Recorded-only provenance: the calibrator reads `expert_weights` and
    /// never `selected_experts`, so `top_k` cannot change `latent_telemetry.csv`
    /// — two runs differing only in `top_k` are byte-identical. It is validated
    /// and echoed into `run_manifest.json` to document the intended routing
    /// fan-out for the downstream handoff, not because it steers this run.
    pub top_k: usize,
    pub telemetry: TelemetrySource,
}

/// What a completed run produced.
#[derive(Debug, Clone)]
pub struct SaaqRunReport {
    pub ticks: usize,
    pub latent_csv_path: PathBuf,
    pub latent_csv_sha256: String,
    pub run_manifest_path: PathBuf,
}

// ── CLI entry point ───────────────────────────────────────────────────────

/// Load, validate and execute a `saaq` recipe.
///
/// `output_dir_override` (the CLI's `--output-dir`) wins over the recipe's
/// `outputs.output_dir`; at least one of the two must be present.
pub fn run_saaq_command(
    recipe_path: &Path,
    output_dir_override: Option<&Path>,
) -> Result<String, String> {
    let config = SaaqRunConfig::load(recipe_path, output_dir_override)?;
    let report = execute(&config)?;

    Ok(format!(
        "✓ SAAQ run '{}' complete ({} ticks, projection {}, rule {}{})\n  \
         latent telemetry: {}\n  sha256:           {}\n  run manifest:     {}",
        config.recipe_id,
        report.ticks,
        projection_mode_name(config.projection_mode),
        update_rule_name(config.update_rule),
        if config.dual_rule { ", dual-rule" } else { "" },
        report.latent_csv_path.display(),
        report.latent_csv_sha256,
        report.run_manifest_path.display(),
    ))
}

// ── Recipe loading + validation ───────────────────────────────────────────

impl SaaqRunConfig {
    /// Read a recipe from disk and turn it into a validated run configuration.
    pub fn load(recipe_path: &Path, output_dir_override: Option<&Path>) -> Result<Self, String> {
        let contents = std::fs::read_to_string(recipe_path)
            .map_err(|e| format!("Failed to read recipe '{}': {}", recipe_path.display(), e))?;
        Self::from_json(&contents, recipe_path, output_dir_override)
    }

    /// Turn recipe JSON into a validated run configuration.
    ///
    /// `recipe_path` is used both for error messages and to anchor relative
    /// `inputs.*` references (see [`resolve_input_ref`]).
    pub fn from_json(
        contents: &str,
        recipe_path: &Path,
        output_dir_override: Option<&Path>,
    ) -> Result<Self, String> {
        let raw: RawRecipe = serde_json::from_str(contents)
            .map_err(|e| format!("Failed to parse recipe '{}': {}", recipe_path.display(), e))?;

        let recipe_id = match raw.recipe_id.as_deref() {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => return Err("recipe_id is required".to_string()),
        };

        match raw.recipe_type.as_deref() {
            Some("saaq") => {}
            Some(other) => {
                return Err(format!(
                    "recipe type '{other}' cannot be run by run-saaq; expected type 'saaq'"
                ));
            }
            None => return Err("type is required (expected 'saaq')".to_string()),
        }

        let inputs = raw.inputs.unwrap_or_default();
        let source_manifest = inputs
            .source_manifest
            .as_deref()
            .map(|reference| resolve_input_ref("inputs.source_manifest", reference, recipe_path))
            .transpose()?;
        let goz1_ref = inputs
            .goz1_ref
            .as_deref()
            .map(|reference| resolve_input_ref("inputs.goz1_ref", reference, recipe_path))
            .transpose()?;
        if source_manifest.is_none() && goz1_ref.is_none() {
            return Err(
                "saaq recipes require inputs.source_manifest or inputs.goz1_ref".to_string(),
            );
        }

        let outputs = raw.outputs.unwrap_or_default();
        let output_dir = match (output_dir_override, outputs.output_dir.as_deref()) {
            (Some(path), _) => path.to_path_buf(),
            (None, Some(dir)) if !dir.is_empty() => PathBuf::from(dir),
            _ => {
                return Err(
                    "outputs.output_dir is required for saaq recipes (or pass --output-dir)"
                        .to_string(),
                );
            }
        };

        let saaq = raw.saaq.unwrap_or_default();

        let projection_mode = match saaq.projection_mode.as_deref() {
            Some(value) => parse_projection_mode(value)?,
            None => ProjectionMode::default(),
        };
        let update_rule = match saaq.update_rule.as_deref() {
            Some(value) => parse_update_rule(value)?,
            None => SaaqUpdateRule::default(),
        };

        let snn_steps = match saaq.snn_steps {
            Some(value) if value >= 1 => value as usize,
            Some(value) => return Err(format!("saaq.snn_steps must be >= 1 (got {value})")),
            None => DEFAULT_SNN_STEPS,
        };

        let thresholds = match saaq.thresholds {
            Some(values) => {
                let len = values.len();
                let array: [f32; 4] = values.try_into().map_err(|_| {
                    format!(
                        "saaq.thresholds must contain exactly 4 numbers, one per telemetry \
                         channel (gpu_temp_c, gpu_power_w, cpu_tctl_c, cpu_package_power_w); \
                         got {len}"
                    )
                })?;
                if let Some(bad) = array
                    .iter()
                    .find(|value| !value.is_finite() || **value <= 0.0)
                {
                    return Err(format!(
                        "saaq.thresholds entries must be finite and > 0 (got {bad})"
                    ));
                }
                array
            }
            None => DEFAULT_THRESHOLDS,
        };

        let num_experts = match saaq.num_experts {
            Some(value) if value >= 2 && value <= magere_corinth_core::EMBEDDING_DIM as i64 => {
                value as usize
            }
            Some(value) if value < 2 => {
                return Err(format!(
                    "saaq.num_experts must be >= 2 for routing entropy to be defined (got {value})"
                ));
            }
            Some(value) => {
                // Slices are `ceil(EMBEDDING_DIM / num_experts)` wide, so beyond
                // one position per expert the trailing experts get empty slices
                // that score exactly 0.0, tie in the softmax, and drag entropy
                // toward its uniform maximum — inflating `routing_entropy`
                // without adding signal. Capping here also bounds the
                // allocation the value drives.
                return Err(format!(
                    "saaq.num_experts must be <= the projector embedding dim ({}) so every \
                     expert owns at least one embedding position; got {value}",
                    magere_corinth_core::EMBEDDING_DIM
                ));
            }
            None => DEFAULT_NUM_EXPERTS,
        };

        let top_k = match saaq.top_k {
            Some(value) if value >= 1 && (value as usize) <= num_experts => value as usize,
            Some(value) => {
                return Err(format!(
                    "saaq.top_k must be between 1 and saaq.num_experts ({num_experts}); got {value}"
                ));
            }
            None => DEFAULT_TOP_K.min(num_experts),
        };

        // `run-saaq` reads the recipe directly rather than through
        // recipe.schema.json, so the schema's enum is not otherwise enforced
        // here and an unsupported value would be copied verbatim into the run
        // manifest's handoff record.
        if let Some(format) = outputs.generated_format.as_deref()
            && !GENERATED_FORMATS.contains(&format)
        {
            return Err(format!(
                "outputs.generated_format must be one of {}; got '{}'",
                GENERATED_FORMATS.join(", "),
                format
            ));
        }

        let telemetry = parse_telemetry(saaq.telemetry.unwrap_or_default(), recipe_path)?;

        Ok(Self {
            recipe_id,
            recipe_path: recipe_path.to_path_buf(),
            description: raw.description,
            source_manifest,
            goz1_ref,
            generated_format: outputs.generated_format,
            manifest_id: outputs.manifest_id,
            output_dir,
            projection_mode,
            snn_steps,
            thresholds,
            update_rule,
            dual_rule: saaq.dual_rule.unwrap_or(false),
            num_experts,
            top_k,
            telemetry,
        })
    }
}

fn parse_projection_mode(value: &str) -> Result<ProjectionMode, String> {
    match value {
        "RateSum" => Ok(ProjectionMode::RateSum),
        "TemporalHistogram" => Ok(ProjectionMode::TemporalHistogram),
        "MembraneSnapshot" => Ok(ProjectionMode::MembraneSnapshot),
        "SpikingTernary" => Ok(ProjectionMode::SpikingTernary),
        other => Err(format!(
            "saaq.projection_mode '{other}' is not supported; expected one of: \
             RateSum, TemporalHistogram, MembraneSnapshot, SpikingTernary"
        )),
    }
}

fn projection_mode_name(mode: ProjectionMode) -> &'static str {
    match mode {
        ProjectionMode::RateSum => "RateSum",
        ProjectionMode::TemporalHistogram => "TemporalHistogram",
        ProjectionMode::MembraneSnapshot => "MembraneSnapshot",
        ProjectionMode::SpikingTernary => "SpikingTernary",
    }
}

fn parse_update_rule(value: &str) -> Result<SaaqUpdateRule, String> {
    match value {
        "LegacyV1_0" => Ok(SaaqUpdateRule::LegacyV1_0),
        "SaaqV1_5SqrtRate" => Ok(SaaqUpdateRule::SaaqV1_5SqrtRate),
        other => Err(format!(
            "saaq.update_rule '{other}' is not supported; expected one of: \
             LegacyV1_0, SaaqV1_5SqrtRate"
        )),
    }
}

fn update_rule_name(rule: SaaqUpdateRule) -> &'static str {
    match rule {
        SaaqUpdateRule::LegacyV1_0 => "LegacyV1_0",
        SaaqUpdateRule::SaaqV1_5SqrtRate => "SaaqV1_5SqrtRate",
    }
}

fn parse_telemetry(raw: RawTelemetry, recipe_path: &Path) -> Result<TelemetrySource, String> {
    let source = raw.source.as_deref().unwrap_or("synthetic");
    match source {
        "synthetic" => {
            let ticks = match raw.ticks {
                Some(value) if value >= 1 => value as usize,
                Some(value) => {
                    return Err(format!("saaq.telemetry.ticks must be >= 1 (got {value})"));
                }
                None => DEFAULT_TICKS,
            };
            let tick_interval_ms = match raw.tick_interval_ms {
                Some(value) if value >= 1 => value as u64,
                Some(value) => {
                    return Err(format!(
                        "saaq.telemetry.tick_interval_ms must be >= 1 (got {value})"
                    ));
                }
                None => DEFAULT_TICK_INTERVAL_MS,
            };
            let start_timestamp_ms = match raw.start_timestamp_ms {
                Some(value) if value >= 0 => value as u64,
                Some(value) => {
                    return Err(format!(
                        "saaq.telemetry.start_timestamp_ms must be >= 0 (got {value})"
                    ));
                }
                None => 0,
            };
            // Each field is individually valid, but the tick-0..tick-(n-1)
            // timestamp ramp they describe need not fit in a u64: a debug build
            // would panic on the overflow and a release build would wrap into
            // non-monotonic timestamps, collapsing the calibrator's time window
            // to its 1 ms fallback. Reject the recipe instead.
            let last_tick = (ticks - 1) as u64;
            if tick_interval_ms
                .checked_mul(last_tick)
                .and_then(|span| start_timestamp_ms.checked_add(span))
                .is_none()
            {
                return Err(format!(
                    "saaq.telemetry timestamps overflow u64: start_timestamp_ms \
                     ({start_timestamp_ms}) + tick_interval_ms ({tick_interval_ms}) * \
                     {last_tick} exceeds u64::MAX"
                ));
            }
            if raw.path.is_some() {
                return Err(
                    "saaq.telemetry.path is only valid for source 'csv', not 'synthetic'"
                        .to_string(),
                );
            }
            let start = raw.start.unwrap_or(Channels {
                gpu_temp_c: 58.0,
                gpu_power_w: 240.0,
                cpu_tctl_c: 62.0,
                cpu_package_power_w: 95.0,
            });
            let delta = raw.delta.unwrap_or(Channels {
                gpu_temp_c: 0.6,
                gpu_power_w: 3.0,
                cpu_tctl_c: 0.45,
                cpu_package_power_w: 2.5,
            });
            for (label, channels) in [("start", &start), ("delta", &delta)] {
                if !channels_are_finite(channels) {
                    return Err(format!(
                        "saaq.telemetry.{label} channel values must all be finite"
                    ));
                }
            }
            // Finite `start` and `delta` do not make the ramp finite: with both
            // at 3e38, tick 1 is already infinity, which would be stored as the
            // encoder baseline and written to the latent CSV. The last tick has
            // the largest magnitude, so checking it covers the whole ramp — and
            // keeps the synthetic source symmetric with the finiteness the CSV
            // source now enforces on the values it reads.
            let last = synthetic_snapshot(
                &SyntheticTelemetry {
                    ticks,
                    tick_interval_ms,
                    start_timestamp_ms,
                    start,
                    delta,
                },
                ticks - 1,
            );
            if !last.gpu_temp_c.is_finite()
                || !last.gpu_power_w.is_finite()
                || !last.cpu_tctl_c.is_finite()
                || !last.cpu_package_power_w.is_finite()
            {
                return Err(format!(
                    "saaq.telemetry ramp overflows to a non-finite value by tick {}: \
                     each channel's start + delta * tick must stay finite",
                    ticks - 1
                ));
            }
            Ok(TelemetrySource::Synthetic(SyntheticTelemetry {
                ticks,
                tick_interval_ms,
                start_timestamp_ms,
                start,
                delta,
            }))
        }
        "csv" => {
            // Synthetic-only knobs are rejected rather than silently ignored,
            // so a recipe never claims a shape the run did not use.
            for (field, present) in [
                ("ticks", raw.ticks.is_some()),
                ("tick_interval_ms", raw.tick_interval_ms.is_some()),
                ("start_timestamp_ms", raw.start_timestamp_ms.is_some()),
                ("start", raw.start.is_some()),
                ("delta", raw.delta.is_some()),
            ] {
                if present {
                    return Err(format!(
                        "saaq.telemetry.{field} is only valid for source 'synthetic'; \
                         source 'csv' takes its ticks from the file at saaq.telemetry.path"
                    ));
                }
            }
            let reference = raw
                .path
                .as_deref()
                .filter(|value| !value.is_empty())
                .ok_or(
                    "saaq.telemetry.path is required when saaq.telemetry.source is 'csv'"
                        .to_string(),
                )?;
            let resolved =
                resolve_input_ref("saaq.telemetry.path", reference, recipe_path)?.resolved_path;
            // Parse eagerly so a malformed CSV fails validation, not mid-run,
            // and keep the rows so execution never re-reads the file.
            let (snapshots, sha256) = read_telemetry_csv(&resolved)?;
            Ok(TelemetrySource::Csv {
                path: resolved,
                snapshots,
                sha256,
            })
        }
        other => Err(format!(
            "saaq.telemetry.source '{other}' is not supported; expected one of: synthetic, csv"
        )),
    }
}

fn channels_are_finite(channels: &Channels) -> bool {
    channels.gpu_temp_c.is_finite()
        && channels.gpu_power_w.is_finite()
        && channels.cpu_tctl_c.is_finite()
        && channels.cpu_package_power_w.is_finite()
}

/// Files that mark the root of the tree a recipe is allowed to reference.
const REPO_ROOT_MARKERS: [&str; 3] = ["Cargo.toml", ".git", "schemas"];

/// Resolve a recipe input reference to an existing file.
///
/// Recipes are checked in with repo-root-relative references
/// (`manifests/examples/...`) but are themselves nested (`configs/recipes/...`),
/// so the reference is tried against the recipe's own directory and then each
/// ancestor **up to and including the first one that looks like a repo root**
/// (see [`REPO_ROOT_MARKERS`]). An absolute reference is taken as written.
///
/// The walk stops at the repo root deliberately: an unbounded walk to `/` let a
/// reference like `etc/hostname` or `../../../etc/passwd` resolve to a real
/// system file and be recorded as the run's provenance. Relative references
/// containing `..` are rejected, and canonicalized candidates must stay under
/// that root, so neither parent components nor symlinks can escape the tree.
///
/// The resolved path and its digest are captured together during validation, so
/// the run manifest names and pins the file that was actually accepted even if
/// the path changes before the CPU-heavy run finishes.
fn resolve_input_ref(
    field: &str,
    reference: &str,
    recipe_path: &Path,
) -> Result<ResolvedRef, String> {
    if reference.is_empty() {
        return Err(format!("{field} must not be empty"));
    }

    let resolved = |candidate: PathBuf| -> Result<ResolvedRef, String> {
        let resolved_path = candidate.canonicalize().map_err(|e| {
            format!(
                "Failed to canonicalize {field} '{}' at '{}': {e}",
                reference,
                candidate.display()
            )
        })?;
        let sha256 = checksum::compute_file_sha256(&resolved_path).map_err(|e| {
            format!(
                "Failed to checksum {field} '{}' at '{}': {e}",
                reference,
                resolved_path.display()
            )
        })?;
        Ok(ResolvedRef {
            reference: reference.to_string(),
            resolved_path,
            sha256,
        })
    };

    let direct = PathBuf::from(reference);
    if direct.is_absolute() {
        if direct.is_file() {
            return resolved(direct);
        }
        return Err(format!("{field} '{reference}' does not resolve to a file"));
    }

    if direct
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(format!(
            "{field} '{reference}' must not contain '..' segments; relative recipe inputs are resolved within the repository that contains the recipe"
        ));
    }

    // A checked-in recipe searches only its containing repository. A marker-less
    // tree is bounded to the recipe directory itself rather than walking to `/`.
    let recipe_dir = recipe_path.parent().unwrap_or(Path::new("."));
    let boundary = recipe_dir
        .ancestors()
        .find(|ancestor| {
            REPO_ROOT_MARKERS
                .iter()
                .any(|marker| ancestor.join(marker).exists())
        })
        .unwrap_or(recipe_dir);
    let canonical_boundary = boundary.canonicalize().map_err(|e| {
        format!(
            "Failed to canonicalize recipe reference boundary '{}': {e}",
            boundary.display()
        )
    })?;

    for ancestor in recipe_dir
        .ancestors()
        .take_while(|ancestor| *ancestor != boundary)
        .chain(std::iter::once(boundary))
    {
        let candidate = ancestor.join(&direct);
        if candidate.is_file() {
            let canonical_candidate = candidate.canonicalize().map_err(|e| {
                format!(
                    "Failed to canonicalize {field} '{}' at '{}': {e}",
                    reference,
                    candidate.display()
                )
            })?;
            if !canonical_candidate.starts_with(&canonical_boundary) {
                return Err(format!(
                    "{field} '{reference}' resolves outside the recipe's allowed tree '{}': '{}'",
                    canonical_boundary.display(),
                    canonical_candidate.display()
                ));
            }
            return resolved(canonical_candidate);
        }
    }

    Err(format!(
        "{field} '{reference}' does not resolve to a file within the recipe's tree (looked relative to '{}' up to '{}')",
        recipe_dir.display(),
        canonical_boundary.display()
    ))
}

// ── Telemetry sources ─────────────────────────────────────────────────────

/// Synthesise the snapshot for `tick` — a pure function of the tick index.
fn synthetic_snapshot(synthetic: &SyntheticTelemetry, tick: usize) -> TelemetrySnapshot {
    let step = tick as f32;
    TelemetrySnapshot {
        gpu_temp_c: synthetic.start.gpu_temp_c + synthetic.delta.gpu_temp_c * step,
        gpu_power_w: synthetic.start.gpu_power_w + synthetic.delta.gpu_power_w * step,
        cpu_tctl_c: synthetic.start.cpu_tctl_c + synthetic.delta.cpu_tctl_c * step,
        cpu_package_power_w: synthetic.start.cpu_package_power_w
            + synthetic.delta.cpu_package_power_w * step,
        timestamp_ms: synthetic.start_timestamp_ms + synthetic.tick_interval_ms * tick as u64,
    }
}

/// Read a telemetry CSV into snapshots.
///
/// The header names the columns (order is free) and must contain every entry of
/// [`TELEMETRY_CSV_COLUMNS`]; extra columns are ignored. Blank lines are
/// skipped so a trailing newline is not an error.
fn read_telemetry_csv(path: &Path) -> Result<(Vec<TelemetrySnapshot>, String), String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read telemetry CSV '{}': {}", path.display(), e))?;
    // Hash the same in-memory bytes parsed below. Re-reading the path after the
    // CPU-heavy run would let an appending/replaced capture make the manifest
    // name bytes that never produced the retained snapshots.
    let sha256 = checksum::compute_string_sha256(&contents);

    // Line numbers are captured before blank lines are filtered so an error
    // points at the line the reader would find in an editor, not at an index
    // into the surviving data rows.
    let mut lines = contents
        .lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty());

    let (_, header) = lines
        .next()
        .ok_or_else(|| format!("Telemetry CSV '{}' is empty", path.display()))?;
    let columns: Vec<&str> = header.split(',').map(str::trim).collect();

    // A duplicated required column would otherwise silently bind to whichever
    // copy came first, making the run depend on column order.
    for name in TELEMETRY_CSV_COLUMNS {
        if columns.iter().filter(|column| **column == name).count() > 1 {
            return Err(format!(
                "Telemetry CSV '{}' declares column '{}' more than once",
                path.display(),
                name
            ));
        }
    }

    let mut indices = [0usize; TELEMETRY_CSV_COLUMNS.len()];
    for (slot, name) in indices.iter_mut().zip(TELEMETRY_CSV_COLUMNS) {
        *slot = columns
            .iter()
            .position(|column| *column == name)
            .ok_or_else(|| {
                format!(
                    "Telemetry CSV '{}' is missing required column '{}' (needs: {})",
                    path.display(),
                    name,
                    TELEMETRY_CSV_COLUMNS.join(", ")
                )
            })?;
    }

    let mut snapshots = Vec::new();
    for (line_no, line) in lines {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let field = |index: usize, name: &str| -> Result<&str, String> {
            fields.get(index).copied().ok_or_else(|| {
                format!(
                    "Telemetry CSV '{}' line {} has no value for column '{}'",
                    path.display(),
                    line_no,
                    name
                )
            })
        };
        let number = |index: usize, name: &str| -> Result<f32, String> {
            let raw = field(index, name)?;
            let value = raw.parse::<f32>().map_err(|e| {
                format!(
                    "Telemetry CSV '{}' line {} column '{}': {}",
                    path.display(),
                    line_no,
                    name,
                    e
                )
            })?;
            // `"NaN"` and `"inf"` parse cleanly as f32. Reject them here so CSV
            // replay holds the same finiteness guarantee the synthetic source
            // and `saaq.thresholds` already enforce, instead of letting
            // non-finite values reach the encoder and the emitted latent CSV.
            if !value.is_finite() {
                return Err(format!(
                    "Telemetry CSV '{}' line {} column '{}': value must be finite (got '{}')",
                    path.display(),
                    line_no,
                    name,
                    raw
                ));
            }
            Ok(value)
        };

        let timestamp_ms = field(indices[0], TELEMETRY_CSV_COLUMNS[0])?
            .parse::<u64>()
            .map_err(|e| {
                format!(
                    "Telemetry CSV '{}' line {} column '{}': {}",
                    path.display(),
                    line_no,
                    TELEMETRY_CSV_COLUMNS[0],
                    e
                )
            })?;

        snapshots.push(TelemetrySnapshot {
            timestamp_ms,
            gpu_temp_c: number(indices[1], TELEMETRY_CSV_COLUMNS[1])?,
            gpu_power_w: number(indices[2], TELEMETRY_CSV_COLUMNS[2])?,
            cpu_tctl_c: number(indices[3], TELEMETRY_CSV_COLUMNS[3])?,
            cpu_package_power_w: number(indices[4], TELEMETRY_CSV_COLUMNS[4])?,
        });
    }

    if snapshots.is_empty() {
        return Err(format!(
            "Telemetry CSV '{}' has a header but no data rows",
            path.display()
        ));
    }

    // `SnnLatentCalibrator::window_dt_ms` only measures a real window when the
    // timestamp advances; a duplicate or backwards row silently falls back to a
    // 1 ms window, which inflates the firing-rate and membrane-derivative
    // columns instead of reporting the malformed capture. Reject it up front so
    // concatenated or unsorted captures fail loudly.
    for pair in snapshots.windows(2) {
        if pair[1].timestamp_ms <= pair[0].timestamp_ms {
            return Err(format!(
                "Telemetry CSV '{}' timestamps must strictly increase: row with \
                 timestamp_ms {} does not advance past the preceding {}",
                path.display(),
                pair[1].timestamp_ms,
                pair[0].timestamp_ms
            ));
        }
    }

    Ok((snapshots, sha256))
}

// ── Deterministic routing surrogate ───────────────────────────────────────

/// Derive expert weights from a projector embedding.
///
/// The embedding is cut into `num_experts` contiguous, equal-width slices
/// (`chunk = ceil(len / num_experts)`, trailing experts get whatever is left,
/// possibly nothing). Each expert scores the mean of its slice, and the scores
/// become a distribution via a max-subtracted softmax.
///
/// This is a stand-in for a real MoE router: `magere-corinth-core`'s SAAQ
/// calibrator only needs a routing *distribution* to compute
/// `routing_entropy`, and this repository does not execute model weights. The
/// function is pure — same embedding in, same weights out — which is what keeps
/// `latent_telemetry.csv` reproducible.
fn expert_weights_from_embedding(embedding: &[f32], num_experts: usize) -> Vec<f32> {
    let chunk = embedding.len().div_ceil(num_experts.max(1)).max(1);
    let scores: Vec<f32> = (0..num_experts)
        .map(|expert| {
            let start = (expert * chunk).min(embedding.len());
            let end = (start + chunk).min(embedding.len());
            let slice = &embedding[start..end];
            if slice.is_empty() {
                0.0
            } else {
                slice.iter().sum::<f32>() / slice.len() as f32
            }
        })
        .collect();
    softmax(&scores)
}

/// Max-subtracted softmax; falls back to a uniform distribution if the inputs
/// are degenerate (empty or non-finite).
fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let uniform = || vec![1.0 / scores.len() as f32; scores.len()];

    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return uniform();
    }
    let exponentials: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
    let total: f32 = exponentials.iter().sum();
    if total > 0.0 && total.is_finite() {
        exponentials.iter().map(|value| value / total).collect()
    } else {
        uniform()
    }
}

/// The `top_k` heaviest experts, heaviest first, ties broken by ascending index.
fn select_experts(weights: &[f32], top_k: usize) -> Vec<usize> {
    let mut ranked: Vec<usize> = (0..weights.len()).collect();
    ranked.sort_by(|&left, &right| {
        weights[right]
            .partial_cmp(&weights[left])
            .unwrap_or(Ordering::Equal)
            .then(left.cmp(&right))
    });
    ranked.truncate(top_k.min(weights.len()));
    ranked
}

/// Per-neuron mean spikes-per-step over the tick's hidden spike train.
fn firing_rates(spike_train: &[Vec<usize>], neurons: usize) -> Vec<f32> {
    let mut rates = vec![0.0_f32; neurons];
    for step in spike_train {
        for &index in step {
            if index < neurons {
                rates[index] += 1.0;
            }
        }
    }
    let steps = spike_train.len().max(1) as f32;
    for rate in &mut rates {
        *rate /= steps;
    }
    rates
}

// ── Execution ─────────────────────────────────────────────────────────────

/// Either calibrator flavour, behind one `observe` call.
enum Calibrator {
    Solo(Box<SnnLatentCalibrator>),
    Dual(Box<SnnDualLatentCalibrator>),
}

impl Calibrator {
    fn new(config: &SaaqRunConfig) -> Self {
        if config.dual_rule {
            Self::Dual(Box::new(SnnDualLatentCalibrator::new(config.update_rule)))
        } else {
            Self::Solo(Box::new(SnnLatentCalibrator::with_update_rule(
                config.update_rule,
            )))
        }
    }

    fn observe(
        &mut self,
        snapshot: &TelemetrySnapshot,
        activity: &magere_corinth_core::FunnelActivity,
        output: &ModelOutput,
    ) -> Result<SnnLatentSnapshot, String> {
        let result = match self {
            Self::Solo(calibrator) => calibrator.observe(snapshot, activity, output),
            Self::Dual(calibrator) => calibrator.observe(snapshot, activity, output),
        };
        result.map_err(|e| format!("SAAQ calibration failed: {e}"))
    }
}

/// Run the pipeline and write `latent_telemetry.csv` + `run_manifest.json`.
pub fn execute(config: &SaaqRunConfig) -> Result<SaaqRunReport, String> {
    let snapshots = match &config.telemetry {
        TelemetrySource::Synthetic(synthetic) => (0..synthetic.ticks)
            .map(|tick| synthetic_snapshot(synthetic, tick))
            .collect::<Vec<_>>(),
        TelemetrySource::Csv { snapshots, .. } => snapshots.clone(),
    };

    std::fs::create_dir_all(&config.output_dir).map_err(|e| {
        format!(
            "Failed to create output directory '{}': {}",
            config.output_dir.display(),
            e
        )
    })?;

    let latent_csv_path = config.output_dir.join(LATENT_CSV_FILE);
    let run_manifest_path = config.output_dir.join(RUN_MANIFEST_FILE);

    // The emitted latent CSV carries every telemetry input column (plus the
    // derived ones the reader ignores), and any numeric CSV can be named
    // `run_manifest.json`. Replaying either output path in place would destroy
    // the source and leave provenance pointing at newly written output bytes.
    if let TelemetrySource::Csv { path, .. } = &config.telemetry {
        let input = path.canonicalize().map_err(|e| {
            format!(
                "Failed to canonicalize saaq.telemetry.path '{}': {e}",
                path.display()
            )
        })?;
        let output_dir = config.output_dir.canonicalize().map_err(|e| {
            format!(
                "Failed to canonicalize output directory '{}': {e}",
                config.output_dir.display()
            )
        })?;
        let collision = [&latent_csv_path, &run_manifest_path]
            .into_iter()
            .find_map(|output| {
                let canonical_output = output.canonicalize().unwrap_or_else(|_| {
                    output_dir.join(
                        output
                            .file_name()
                            .expect("run output paths always have a file name"),
                    )
                });
                (input == canonical_output).then_some(canonical_output)
            });
        if let Some(output) = collision {
            return Err(format!(
                "saaq.telemetry.path '{}' resolves to this run's own output '{}'; \
                 pick a different --output-dir so the replay does not overwrite its source",
                path.display(),
                output.display()
            ));
        }
    }

    let mut funnel = TelemetryFunnel::new(config.thresholds, config.snn_steps);
    let mut projector =
        Projector::with_input_neurons(config.projection_mode, FUNNEL_HIDDEN_NEURONS);
    let mut calibrator = Calibrator::new(config);
    let mut exporter = SnnLatentCsvExporter::create(&latent_csv_path)
        .map_err(|e| format!("Failed to create '{}': {}", latent_csv_path.display(), e))?;

    for snapshot in &snapshots {
        let activity = funnel.encode_snapshot(snapshot);
        let embedding = projector
            .project(
                &activity.spike_train,
                &activity.potentials,
                &activity.iz_potentials,
            )
            .map_err(|e| format!("Projection failed: {e}"))?;

        let expert_weights = expert_weights_from_embedding(&embedding, config.num_experts);
        let selected_experts = select_experts(&expert_weights, config.top_k);
        let output = ModelOutput {
            spike_train: activity.spike_train.clone(),
            firing_rates: firing_rates(&activity.spike_train, FUNNEL_HIDDEN_NEURONS),
            membrane_potentials: activity.potentials.clone(),
            embedding,
            expert_weights: Some(expert_weights),
            selected_experts: Some(selected_experts),
            reasoning: None,
        };

        let latent = calibrator.observe(snapshot, &activity, &output)?;
        exporter
            .write_row(&latent)
            .map_err(|e| format!("Failed to write '{}': {}", latent_csv_path.display(), e))?;
    }

    exporter
        .flush()
        .map_err(|e| format!("Failed to flush '{}': {}", latent_csv_path.display(), e))?;
    drop(exporter);

    let latent_csv_sha256 = checksum::compute_file_sha256(&latent_csv_path)
        .map_err(|e| format!("Failed to checksum '{}': {}", latent_csv_path.display(), e))?;

    let manifest = build_run_manifest(config, &snapshots, &latent_csv_path, &latent_csv_sha256);
    let serialized = serde_json::to_string_pretty(&manifest)
        .map_err(|e| format!("Failed to serialize run manifest: {e}"))?;
    std::fs::write(&run_manifest_path, format!("{serialized}\n"))
        .map_err(|e| format!("Failed to write '{}': {}", run_manifest_path.display(), e))?;

    Ok(SaaqRunReport {
        ticks: snapshots.len(),
        latent_csv_path,
        latent_csv_sha256,
        run_manifest_path,
    })
}

/// Render an `f32` as JSON without f64-widening artefacts: `0.45_f32 as f64`
/// prints as `0.44999998807907104`, whereas `f32`'s `Display` already emits the
/// shortest decimal that round-trips, so re-parsing it as `f64` echoes the
/// recipe's own literal back.
fn json_f32(value: f32) -> serde_json::Value {
    format!("{value}")
        .parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map_or(serde_json::Value::Null, serde_json::Value::Number)
}

fn channels_json(channels: &Channels) -> serde_json::Value {
    serde_json::json!({
        "gpu_temp_c": json_f32(channels.gpu_temp_c),
        "gpu_power_w": json_f32(channels.gpu_power_w),
        "cpu_tctl_c": json_f32(channels.cpu_tctl_c),
        "cpu_package_power_w": json_f32(channels.cpu_package_power_w),
    })
}

fn build_run_manifest(
    config: &SaaqRunConfig,
    snapshots: &[TelemetrySnapshot],
    latent_csv_path: &Path,
    latent_csv_sha256: &str,
) -> serde_json::Value {
    let reference = |resolved: &Option<ResolvedRef>| -> serde_json::Value {
        match resolved {
            Some(resolved) => serde_json::json!({
                "ref": resolved.reference,
                "resolved_path": resolved.resolved_path.display().to_string(),
                // The path is mutable, so use the digest retained when
                // validation accepted it rather than re-reading later bytes.
                "sha256": resolved.sha256,
            }),
            None => serde_json::Value::Null,
        }
    };

    let telemetry = match &config.telemetry {
        TelemetrySource::Synthetic(synthetic) => serde_json::json!({
            "source": "synthetic",
            "ticks": synthetic.ticks,
            "tick_interval_ms": synthetic.tick_interval_ms,
            "start_timestamp_ms": synthetic.start_timestamp_ms,
            "start": channels_json(&synthetic.start),
            "delta": channels_json(&synthetic.delta),
            "note": "channel c at tick i = start.c + delta.c * i; timestamps are recipe-derived, not wall-clock",
        }),
        TelemetrySource::Csv { path, sha256, .. } => serde_json::json!({
            "source": "csv",
            "path": path.display().to_string(),
            // The path alone is a mutable reference: without the digest of the
            // bytes that actually drove this run, the manifest cannot be used
            // to verify the run was reproduced from the same input.
            "sha256": sha256,
            "ticks": snapshots.len(),
        }),
    };

    serde_json::json!({
        "schema": RUN_MANIFEST_SCHEMA,
        "created_at": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "recipe": {
            "recipe_id": config.recipe_id,
            "type": "saaq",
            "path": config.recipe_path.display().to_string(),
            "description": config.description,
        },
        "inputs": {
            "source_manifest": reference(&config.source_manifest),
            "goz1_ref": reference(&config.goz1_ref),
        },
        "saaq": {
            "projection_mode": projection_mode_name(config.projection_mode),
            "snn_steps": config.snn_steps,
            "thresholds": config.thresholds.iter().copied().map(json_f32).collect::<Vec<_>>(),
            "update_rule": update_rule_name(config.update_rule),
            "dual_rule": config.dual_rule,
            "num_experts": config.num_experts,
            "top_k": config.top_k,
            "hidden_neurons": FUNNEL_HIDDEN_NEURONS,
            "embedding_dim": magere_corinth_core::EMBEDDING_DIM,
            "expert_weight_scheme": EXPERT_WEIGHT_SCHEME,
        },
        "telemetry": telemetry,
        "ticks": snapshots.len(),
        "outputs": {
            "output_dir": config.output_dir.display().to_string(),
            "generated_format": config.generated_format,
            "manifest_id": config.manifest_id,
            "latent_telemetry_csv": latent_csv_path.display().to_string(),
            "latent_telemetry_sha256": latent_csv_sha256,
            "rows": snapshots.len(),
        },
        "crate_versions": {
            "magere-cli": env!("CARGO_PKG_VERSION"),
            "magere-corinth-core": magere_corinth_core::CRATE_VERSION,
        },
        "determinism": {
            "cpu_only": true,
            "wall_clock_in_csv": false,
            "note": "latent_telemetry.csv is a pure function of this recipe; created_at is the only wall-clock value and lives here, not in the CSV",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    const CSV_HEADER: &str = "timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,saaq_delta_q_prev,saaq_delta_q_target,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w,saaq_delta_q_legacy_prev,saaq_delta_q_legacy_target,saaq_delta_q_v15_prev,saaq_delta_q_v15_target";

    fn repo_root() -> PathBuf {
        // crates/magere-cli -> crates -> <repo root>
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("magere-cli lives two levels below the repo root")
            .to_path_buf()
    }

    fn example_manifest() -> PathBuf {
        repo_root().join("manifests/examples/olmoe-1b-7b-instruct.json")
    }

    /// A minimal but complete `saaq` recipe; `extra` is spliced into the
    /// `saaq` block so individual tests can override single fields.
    fn recipe_json(ticks: usize, extra: &str) -> String {
        format!(
            r#"{{
  "recipe_id": "unit-test-saaq",
  "type": "saaq",
  "inputs": {{ "source_manifest": "{manifest}" }},
  "saaq": {{
    "projection_mode": "SpikingTernary",
    "snn_steps": 8,
    "thresholds": [1.0, 5.0, 1.0, 5.0],
    "update_rule": "SaaqV1_5SqrtRate",
    "dual_rule": true,
    "num_experts": 8,
    "top_k": 2,
    "telemetry": {{
      "source": "synthetic",
      "ticks": {ticks},
      "tick_interval_ms": 250,
      "start_timestamp_ms": 0,
      "start": {{ "gpu_temp_c": 58.0, "gpu_power_w": 240.0, "cpu_tctl_c": 62.0, "cpu_package_power_w": 95.0 }},
      "delta": {{ "gpu_temp_c": 0.6, "gpu_power_w": 3.0, "cpu_tctl_c": 0.45, "cpu_package_power_w": 2.5 }}
    }}{extra}
  }}
}}"#,
            manifest = example_manifest().display(),
        )
    }

    fn config_from(json: &str, output_dir: &Path) -> Result<SaaqRunConfig, String> {
        SaaqRunConfig::from_json(
            json,
            &repo_root().join("configs/recipes/unit-test.json"),
            Some(output_dir),
        )
    }

    fn run(json: &str, output_dir: &Path) -> Result<SaaqRunReport, String> {
        execute(&config_from(json, output_dir)?)
    }

    // ── Output shape ──────────────────────────────────────────────────────

    #[test]
    fn csv_header_matches_exporter_and_row_count_matches_ticks() {
        let ticks = 4;
        let out = TempDir::new().unwrap();
        let report = run(&recipe_json(ticks, ""), out.path()).unwrap();

        assert_eq!(report.ticks, ticks);

        let contents = std::fs::read_to_string(&report.latent_csv_path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines[0], CSV_HEADER);
        assert_eq!(
            lines.len() - 1,
            ticks,
            "expected one data row per tick, got {} rows",
            lines.len() - 1
        );
        // Every row must carry all 14 SnnLatentSnapshot fields.
        for row in &lines[1..] {
            assert_eq!(
                row.split(',').count(),
                14,
                "unexpected column count in {row}"
            );
        }
    }

    #[test]
    fn run_manifest_records_effective_parameters() {
        let out = TempDir::new().unwrap();
        let report = run(&recipe_json(3, ""), out.path()).unwrap();

        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&report.run_manifest_path).unwrap())
                .unwrap();

        assert_eq!(manifest["schema"], RUN_MANIFEST_SCHEMA);
        assert_eq!(manifest["recipe"]["recipe_id"], "unit-test-saaq");
        assert_eq!(manifest["ticks"], 3);
        assert_eq!(manifest["saaq"]["projection_mode"], "SpikingTernary");
        assert_eq!(manifest["saaq"]["update_rule"], "SaaqV1_5SqrtRate");
        assert_eq!(manifest["saaq"]["snn_steps"], 8);
        assert_eq!(manifest["saaq"]["dual_rule"], true);
        assert_eq!(manifest["saaq"]["num_experts"], 8);
        assert_eq!(manifest["saaq"]["top_k"], 2);
        assert_eq!(manifest["telemetry"]["source"], "synthetic");
        assert_eq!(
            manifest["outputs"]["latent_telemetry_sha256"],
            serde_json::Value::String(report.latent_csv_sha256.clone())
        );
        assert!(manifest["inputs"]["source_manifest"]["resolved_path"].is_string());
        assert!(manifest["crate_versions"]["magere-corinth-core"].is_string());
        assert!(manifest["created_at"].is_string());
    }

    // ── Determinism ───────────────────────────────────────────────────────

    #[test]
    fn same_recipe_produces_byte_identical_csv() {
        let json = recipe_json(4, "");
        let first = TempDir::new().unwrap();
        let second = TempDir::new().unwrap();

        let a = run(&json, first.path()).unwrap();
        let b = run(&json, second.path()).unwrap();

        let bytes_a = std::fs::read(&a.latent_csv_path).unwrap();
        let bytes_b = std::fs::read(&b.latent_csv_path).unwrap();
        assert_eq!(
            bytes_a, bytes_b,
            "latent_telemetry.csv must be reproducible"
        );
        assert_eq!(a.latent_csv_sha256, b.latent_csv_sha256);
    }

    #[test]
    fn synthetic_snapshots_are_a_pure_function_of_the_tick_index() {
        let synthetic = SyntheticTelemetry {
            ticks: 3,
            tick_interval_ms: 100,
            start_timestamp_ms: 500,
            start: Channels {
                gpu_temp_c: 60.0,
                gpu_power_w: 250.0,
                cpu_tctl_c: 70.0,
                cpu_package_power_w: 120.0,
            },
            delta: Channels {
                gpu_temp_c: 0.5,
                gpu_power_w: 2.0,
                cpu_tctl_c: 0.25,
                cpu_package_power_w: 1.0,
            },
        };

        let second = synthetic_snapshot(&synthetic, 2);
        assert_eq!(second.timestamp_ms, 700);
        assert_eq!(second.gpu_temp_c, 61.0);
        assert_eq!(second.gpu_power_w, 254.0);
        assert_eq!(second.cpu_tctl_c, 70.5);
        assert_eq!(second.cpu_package_power_w, 122.0);
        // Re-evaluating the same tick yields the same snapshot.
        let again = synthetic_snapshot(&synthetic, 2);
        assert_eq!(second.timestamp_ms, again.timestamp_ms);
        assert_eq!(second.gpu_temp_c, again.gpu_temp_c);
    }

    #[test]
    fn expert_weights_are_a_normalized_pure_function_of_the_embedding() {
        let embedding: Vec<f32> = (0..2048).map(|i| ((i % 7) as f32) - 3.0).collect();
        let first = expert_weights_from_embedding(&embedding, 8);
        let second = expert_weights_from_embedding(&embedding, 8);

        assert_eq!(first, second);
        assert_eq!(first.len(), 8);
        let total: f32 = first.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "weights must sum to 1, got {total}"
        );
        assert!(first.iter().all(|weight| *weight > 0.0));
    }

    #[test]
    fn selected_experts_are_ranked_and_tie_broken_by_index() {
        assert_eq!(select_experts(&[0.1, 0.5, 0.3, 0.1], 2), vec![1, 2]);
        // Equal weights fall back to ascending index.
        assert_eq!(select_experts(&[0.25, 0.25, 0.25, 0.25], 3), vec![0, 1, 2]);
    }

    // ── Validation ────────────────────────────────────────────────────────

    #[test]
    fn rejects_non_saaq_recipe_type() {
        let out = TempDir::new().unwrap();
        let json = recipe_json(2, "").replace("\"type\": \"saaq\"", "\"type\": \"goz1_pack\"");
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(error.contains("goz1_pack"), "unexpected error: {error}");
        assert!(
            error.contains("expected type 'saaq'"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_unknown_projection_mode() {
        let out = TempDir::new().unwrap();
        let json = recipe_json(2, "").replace("\"SpikingTernary\"", "\"RateFusion\"");
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("saaq.projection_mode"),
            "unexpected error: {error}"
        );
        assert!(error.contains("RateFusion"), "unexpected error: {error}");
    }

    #[test]
    fn rejects_unknown_update_rule() {
        let out = TempDir::new().unwrap();
        let json = recipe_json(2, "").replace("\"SaaqV1_5SqrtRate\"", "\"SaaqV9_9\"");
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("saaq.update_rule"),
            "unexpected error: {error}"
        );
        assert!(error.contains("SaaqV9_9"), "unexpected error: {error}");
    }

    #[test]
    fn rejects_thresholds_of_the_wrong_length() {
        let out = TempDir::new().unwrap();
        let json = recipe_json(2, "").replace("[1.0, 5.0, 1.0, 5.0]", "[1.0, 5.0, 1.0]");
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("saaq.thresholds"),
            "unexpected error: {error}"
        );
        assert!(error.contains("exactly 4"), "unexpected error: {error}");
    }

    #[test]
    fn rejects_zero_snn_steps() {
        let out = TempDir::new().unwrap();
        let json = recipe_json(2, "").replace("\"snn_steps\": 8", "\"snn_steps\": 0");
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("saaq.snn_steps"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_input_ref_that_does_not_resolve() {
        let out = TempDir::new().unwrap();
        let json = recipe_json(2, "").replace(
            &example_manifest().display().to_string(),
            "manifests/examples/does-not-exist.json",
        );
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("inputs.source_manifest"),
            "unexpected error: {error}"
        );
        assert!(
            error.contains("does not resolve to a file"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_goz1_ref_that_does_not_resolve() {
        let out = TempDir::new().unwrap();
        let json = recipe_json(2, "").replace(
            "\"inputs\": {",
            "\"inputs\": { \"goz1_ref\": \"packs/nope.goz1\",",
        );
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("inputs.goz1_ref"),
            "unexpected error: {error}"
        );
        assert!(
            error.contains("does not resolve to a file"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_missing_input_refs_entirely() {
        let out = TempDir::new().unwrap();
        let json = recipe_json(2, "").replace(
            &format!(
                "\"inputs\": {{ \"source_manifest\": \"{}\" }},",
                example_manifest().display()
            ),
            "",
        );
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("inputs.source_manifest or inputs.goz1_ref"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_unknown_telemetry_source() {
        let out = TempDir::new().unwrap();
        let json =
            recipe_json(2, "").replace("\"source\": \"synthetic\"", "\"source\": \"prometheus\"");
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("saaq.telemetry.source"),
            "unexpected error: {error}"
        );
        assert!(error.contains("prometheus"), "unexpected error: {error}");
    }

    #[test]
    fn rejects_synthetic_timestamps_that_overflow_u64() {
        let out = TempDir::new().unwrap();
        // Every field is individually valid (ticks >= 1, interval >= 1,
        // start >= 0), but the ramp they describe runs past u64::MAX.
        let json = recipe_json(4, "").replace(
            "\"start_timestamp_ms\": 0",
            "\"start_timestamp_ms\": 9223372036854775807",
        );
        let json = json.replace(
            "\"tick_interval_ms\": 250",
            "\"tick_interval_ms\": 9223372036854775807",
        );
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(error.contains("overflow"), "unexpected error: {error}");
    }

    #[test]
    fn accepts_synthetic_timestamps_at_the_edge_of_u64() {
        let out = TempDir::new().unwrap();
        // The raw fields are i64, so the largest ramp expressible is
        // i64::MAX + i64::MAX * 1 == u64::MAX - 1 at ticks = 2: the last value
        // that must still be accepted.
        let json = recipe_json(2, "")
            .replace(
                "\"start_timestamp_ms\": 0",
                "\"start_timestamp_ms\": 9223372036854775807",
            )
            .replace(
                "\"tick_interval_ms\": 250",
                "\"tick_interval_ms\": 9223372036854775807",
            );
        assert!(config_from(&json, out.path()).is_ok());
    }

    #[test]
    fn rejects_generated_format_outside_the_schema_enum() {
        let out = TempDir::new().unwrap();
        // "awq" is explicitly removed from this repo; it must not reach a run
        // manifest just because run-saaq bypasses the JSON schema.
        let json = recipe_json(2, "").replace(
            "\"recipe_id\": \"unit-test-saaq\"",
            "\"recipe_id\": \"unit-test-saaq\",\n  \"outputs\": { \"generated_format\": \"awq\" }",
        );
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("generated_format"),
            "unexpected error: {error}"
        );
        assert!(error.contains("awq"), "unexpected error: {error}");
    }

    #[test]
    fn accepts_every_schema_generated_format() {
        for format in GENERATED_FORMATS {
            let out = TempDir::new().unwrap();
            let json = recipe_json(2, "").replace(
                "\"recipe_id\": \"unit-test-saaq\"",
                &format!(
                    "\"recipe_id\": \"unit-test-saaq\",\n  \"outputs\": {{ \"generated_format\": \"{format}\" }}"
                ),
            );
            assert!(
                config_from(&json, out.path()).is_ok(),
                "format '{format}' should be accepted"
            );
        }
    }

    #[test]
    fn rejects_num_experts_above_the_embedding_dim() {
        let out = TempDir::new().unwrap();
        // Beyond one embedding position per expert the trailing slices are
        // empty, tie at 0.0, and inflate routing entropy.
        let json = recipe_json(2, "").replace("\"num_experts\": 8", "\"num_experts\": 900000000");
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(error.contains("num_experts"), "unexpected error: {error}");
        assert!(
            error.contains(&magere_corinth_core::EMBEDDING_DIM.to_string()),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn input_refs_do_not_escape_the_repository_tree() {
        let out = TempDir::new().unwrap();
        // An unbounded ancestor walk used to resolve this to /etc/hostname and
        // record a system file as the run's provenance.
        let json = recipe_json(2, "").replace(
            example_manifest().display().to_string().as_str(),
            "etc/hostname",
        );
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("source_manifest"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn input_refs_reject_parent_directory_segments() {
        let out = TempDir::new().unwrap();
        let json = recipe_json(2, "").replace(
            example_manifest().display().to_string().as_str(),
            "../../../../etc/hostname",
        );
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("must not contain '..'"),
            "unexpected error: {error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn input_refs_reject_symlinks_that_escape_the_repository_tree() {
        use std::os::unix::fs::symlink;

        let tree = TempDir::new().unwrap();
        std::fs::write(tree.path().join("Cargo.toml"), "[workspace]\n").unwrap();
        let outside = TempDir::new().unwrap();
        let outside_manifest = outside.path().join("outside.json");
        std::fs::write(&outside_manifest, "outside").unwrap();
        symlink(&outside_manifest, tree.path().join("escape.json")).unwrap();

        let json = recipe_json(2, "").replace(
            example_manifest().display().to_string().as_str(),
            "escape.json",
        );
        let error =
            SaaqRunConfig::from_json(&json, &tree.path().join("recipe.json"), Some(tree.path()))
                .unwrap_err();
        assert!(
            error.contains("outside the recipe's allowed tree"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn top_k_cannot_change_the_emitted_csv() {
        // top_k is recorded-only provenance: the calibrator never reads
        // selected_experts, so the CSV must not depend on it.
        let a = TempDir::new().unwrap();
        let b = TempDir::new().unwrap();
        let one = run(
            &recipe_json(6, "").replace("\"top_k\": 2", "\"top_k\": 1"),
            a.path(),
        )
        .unwrap();
        let many = run(
            &recipe_json(6, "").replace("\"top_k\": 2", "\"top_k\": 8"),
            b.path(),
        )
        .unwrap();
        assert_eq!(one.latent_csv_sha256, many.latent_csv_sha256);
    }

    #[test]
    fn run_manifest_pins_resolved_inputs_by_checksum() {
        let out = TempDir::new().unwrap();
        let report = run(&recipe_json(2, ""), out.path()).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&report.run_manifest_path).unwrap())
                .unwrap();
        let digest = manifest["inputs"]["source_manifest"]["sha256"]
            .as_str()
            .expect("source_manifest must be pinned by sha256")
            .to_string();
        assert_eq!(
            digest.len(),
            64,
            "expected a sha256 hex digest, got {digest}"
        );
        let expected = checksum::compute_file_sha256(example_manifest()).unwrap();
        assert_eq!(digest, expected);
    }

    #[test]
    fn run_manifest_retains_the_provenance_digest_from_validation() {
        let input = TempDir::new().unwrap();
        let source_manifest = input.path().join("source.json");
        let validated_bytes = "{\"version\":1}\n";
        std::fs::write(&source_manifest, validated_bytes).unwrap();

        let out = TempDir::new().unwrap();
        let json = recipe_json(2, "").replace(
            example_manifest().display().to_string().as_str(),
            &source_manifest.display().to_string(),
        );
        let config = config_from(&json, out.path()).unwrap();
        let validated_sha256 = checksum::compute_string_sha256(validated_bytes);

        // Provenance inputs are references, not execution inputs. If one
        // changes during the CPU-heavy run, the manifest must still describe
        // the bytes that validation accepted.
        std::fs::write(&source_manifest, "{\"version\":2}\n").unwrap();
        let report = execute(&config).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&report.run_manifest_path).unwrap())
                .unwrap();
        assert_eq!(
            manifest["inputs"]["source_manifest"]["sha256"],
            validated_sha256
        );
    }

    #[test]
    fn rejects_synthetic_ramps_that_overflow_to_infinity() {
        let out = TempDir::new().unwrap();
        // start and delta are each finite, but tick 1 is already infinity.
        let json = recipe_json(4, "")
            .replace("\"gpu_temp_c\": 58.0", "\"gpu_temp_c\": 3e38")
            .replace("\"gpu_temp_c\": 0.6", "\"gpu_temp_c\": 3e38");
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(error.contains("non-finite"), "unexpected error: {error}");
    }

    #[test]
    fn rejects_missing_output_dir() {
        let json = recipe_json(2, "");
        let error = SaaqRunConfig::from_json(
            &json,
            &repo_root().join("configs/recipes/unit-test.json"),
            None,
        )
        .unwrap_err();
        assert!(
            error.contains("outputs.output_dir"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_unknown_saaq_field() {
        let out = TempDir::new().unwrap();
        let json = recipe_json(2, ",\n    \"unsupported_knob\": 0.1");
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("unsupported_knob"),
            "unexpected error: {error}"
        );
    }

    // ── CSV telemetry source ──────────────────────────────────────────────

    /// A `saaq` recipe whose telemetry comes from a CSV at `telemetry_path`.
    fn csv_recipe_json(telemetry_path: &Path) -> String {
        format!(
            r#"{{
  "recipe_id": "unit-test-saaq-csv",
  "type": "saaq",
  "inputs": {{ "source_manifest": "{manifest}" }},
  "saaq": {{
    "snn_steps": 8,
    "num_experts": 4,
    "telemetry": {{ "source": "csv", "path": "{telemetry}" }}
  }}
}}"#,
            manifest = example_manifest().display(),
            telemetry = telemetry_path.display(),
        )
    }

    #[test]
    fn csv_telemetry_source_replays_rows_in_file_order() {
        let out = TempDir::new().unwrap();
        let telemetry_path = out.path().join("telemetry.csv");
        std::fs::write(
            &telemetry_path,
            "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
             0,58.0,240.0,62.0,95.0\n\
             250,59.2,246.0,62.9,100.0\n\
             500,60.4,252.0,63.8,105.0\n",
        )
        .unwrap();

        let report = run(&csv_recipe_json(&telemetry_path), out.path()).unwrap();
        assert_eq!(report.ticks, 3);

        let contents = std::fs::read_to_string(&report.latent_csv_path).unwrap();
        let rows: Vec<&str> = contents.lines().skip(1).collect();
        assert_eq!(rows.len(), 3);
        assert!(rows[0].starts_with("0,"));
        assert!(rows[1].starts_with("250,"));
        assert!(rows[2].starts_with("500,"));
    }

    #[test]
    fn run_manifest_hashes_the_csv_bytes_that_were_parsed() {
        let out = TempDir::new().unwrap();
        let telemetry_path = out.path().join("telemetry.csv");
        let parsed_bytes = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
             0,58.0,240.0,62.0,95.0\n\
             250,59.2,246.0,62.9,100.0\n";
        std::fs::write(&telemetry_path, parsed_bytes).unwrap();

        let config = config_from(&csv_recipe_json(&telemetry_path), out.path()).unwrap();
        let parsed_sha256 = checksum::compute_string_sha256(parsed_bytes);

        // Simulate a capture process appending after validation. Execution uses
        // the snapshots retained in `config`, so provenance must retain the
        // digest of `parsed_bytes` rather than re-read this later file state.
        std::fs::write(
            &telemetry_path,
            format!("{parsed_bytes}500,60.4,252.0,63.8,105.0\n"),
        )
        .unwrap();
        let later_sha256 = checksum::compute_file_sha256(&telemetry_path).unwrap();
        assert_ne!(parsed_sha256, later_sha256);

        let report = execute(&config).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&report.run_manifest_path).unwrap())
                .unwrap();
        assert_eq!(manifest["telemetry"]["sha256"], parsed_sha256);
        assert_eq!(manifest["telemetry"]["ticks"], 2);
    }

    #[test]
    fn rejects_csv_telemetry_missing_a_required_column() {
        let out = TempDir::new().unwrap();
        let telemetry_path = out.path().join("telemetry.csv");
        std::fs::write(
            &telemetry_path,
            "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c\n0,58.0,240.0,62.0\n",
        )
        .unwrap();

        let error = config_from(&csv_recipe_json(&telemetry_path), out.path()).unwrap_err();
        assert!(
            error.contains("cpu_package_power_w"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_non_finite_csv_telemetry_values() {
        for bad in ["NaN", "inf", "-inf"] {
            let out = TempDir::new().unwrap();
            let telemetry_path = out.path().join("telemetry.csv");
            std::fs::write(
                &telemetry_path,
                format!(
                    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
                     0,58.0,240.0,62.0,95.0\n\
                     250,{bad},246.0,62.9,100.0\n"
                ),
            )
            .unwrap();

            let error = config_from(&csv_recipe_json(&telemetry_path), out.path()).unwrap_err();
            assert!(
                error.contains("must be finite"),
                "unexpected error for '{bad}': {error}"
            );
            assert!(
                error.contains("gpu_temp_c"),
                "unexpected error for '{bad}': {error}"
            );
        }
    }

    #[test]
    fn rejects_non_increasing_csv_timestamps() {
        for rows in [
            // duplicate timestamp
            "0,58.0,240.0,62.0,95.0\n0,59.2,246.0,62.9,100.0\n",
            // backwards timestamp
            "500,58.0,240.0,62.0,95.0\n250,59.2,246.0,62.9,100.0\n",
        ] {
            let out = TempDir::new().unwrap();
            let telemetry_path = out.path().join("telemetry.csv");
            std::fs::write(
                &telemetry_path,
                format!(
                    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n{rows}"
                ),
            )
            .unwrap();

            let error = config_from(&csv_recipe_json(&telemetry_path), out.path()).unwrap_err();
            assert!(
                error.contains("strictly increase"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn rejects_duplicate_csv_columns() {
        let out = TempDir::new().unwrap();
        let telemetry_path = out.path().join("telemetry.csv");
        std::fs::write(
            &telemetry_path,
            "timestamp_ms,gpu_temp_c,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
             0,58.0,99.0,240.0,62.0,95.0\n",
        )
        .unwrap();

        let error = config_from(&csv_recipe_json(&telemetry_path), out.path()).unwrap_err();
        assert!(
            error.contains("more than once"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn csv_errors_report_the_real_file_line() {
        let out = TempDir::new().unwrap();
        let telemetry_path = out.path().join("telemetry.csv");
        // A blank line between the header and the bad row: the bad value sits on
        // file line 4 even though it is only the second data row.
        std::fs::write(
            &telemetry_path,
            "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
             0,58.0,240.0,62.0,95.0\n\
             \n\
             250,not_a_number,246.0,62.9,100.0\n",
        )
        .unwrap();

        let error = config_from(&csv_recipe_json(&telemetry_path), out.path()).unwrap_err();
        assert!(error.contains("line 4"), "unexpected error: {error}");
    }

    #[test]
    fn refuses_to_replay_a_csv_into_its_own_output() {
        let out = TempDir::new().unwrap();
        // A previous run's latent CSV carries every telemetry input column, so
        // it is a valid input; replaying it in place would truncate the source.
        let first = run(&recipe_json(4, ""), out.path()).unwrap();
        let error = SaaqRunConfig::from_json(
            &csv_recipe_json(&first.latent_csv_path),
            &repo_root().join("configs/recipes/unit-test.json"),
            Some(out.path()),
        )
        .and_then(|config| execute(&config).map(|_| ()))
        .unwrap_err();
        assert!(error.contains("own output"), "unexpected error: {error}");
    }

    #[test]
    fn refuses_to_replace_csv_input_named_like_the_run_manifest() {
        let out = TempDir::new().unwrap();
        let telemetry_path = out.path().join(RUN_MANIFEST_FILE);
        let telemetry = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
             0,58.0,240.0,62.0,95.0\n";
        std::fs::write(&telemetry_path, telemetry).unwrap();

        let error = config_from(&csv_recipe_json(&telemetry_path), out.path())
            .and_then(|config| execute(&config).map(|_| ()))
            .unwrap_err();
        assert!(error.contains("own output"), "unexpected error: {error}");
        assert_eq!(
            std::fs::read_to_string(&telemetry_path).unwrap(),
            telemetry,
            "collision detection must run before the source is overwritten"
        );
    }

    #[test]
    fn rejects_csv_telemetry_path_that_does_not_resolve() {
        let out = TempDir::new().unwrap();
        let json = csv_recipe_json(Path::new("telemetry/nope.csv"));
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("saaq.telemetry.path"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_synthetic_only_knobs_under_the_csv_source() {
        let out = TempDir::new().unwrap();
        let telemetry_path = out.path().join("telemetry.csv");
        std::fs::write(
            &telemetry_path,
            "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
             0,58.0,240.0,62.0,95.0\n",
        )
        .unwrap();

        let json = csv_recipe_json(&telemetry_path)
            .replace("\"source\": \"csv\"", "\"source\": \"csv\", \"ticks\": 5");
        let error = config_from(&json, out.path()).unwrap_err();
        assert!(
            error.contains("only valid for source 'synthetic'"),
            "unexpected error: {error}"
        );
    }

    // ── Checked-in example, end to end ────────────────────────────────────

    #[test]
    fn checked_in_example_recipe_runs_end_to_end() {
        let recipe_path = repo_root().join("configs/recipes/saaq-example.json");
        assert!(
            recipe_path.is_file(),
            "missing example recipe at {}",
            recipe_path.display()
        );

        let out = TempDir::new().unwrap();
        let message = run_saaq_command(&recipe_path, Some(out.path())).unwrap();
        assert!(
            message.contains("SAAQ run"),
            "unexpected message: {message}"
        );

        let csv = out.path().join(LATENT_CSV_FILE);
        let manifest_path = out.path().join(RUN_MANIFEST_FILE);
        assert!(csv.is_file(), "missing {}", csv.display());
        assert!(
            manifest_path.is_file(),
            "missing {}",
            manifest_path.display()
        );

        let contents = std::fs::read_to_string(&csv).unwrap();
        let mut lines = contents.lines();
        assert_eq!(lines.next().unwrap(), CSV_HEADER);
        let rows = lines.count();
        assert!(rows > 0, "example recipe produced no telemetry rows");

        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap())
                .expect("run_manifest.json must parse");
        assert_eq!(manifest["recipe"]["type"], "saaq");
        assert_eq!(manifest["ticks"], rows);
        assert_eq!(manifest["outputs"]["rows"], rows);
    }

    #[test]
    fn checked_in_example_recipe_is_reproducible() {
        let recipe_path = repo_root().join("configs/recipes/saaq-example.json");
        let first = TempDir::new().unwrap();
        let second = TempDir::new().unwrap();

        run_saaq_command(&recipe_path, Some(first.path())).unwrap();
        run_saaq_command(&recipe_path, Some(second.path())).unwrap();

        let a = std::fs::read(first.path().join(LATENT_CSV_FILE)).unwrap();
        let b = std::fs::read(second.path().join(LATENT_CSV_FILE)).unwrap();
        assert_eq!(a, b, "the checked-in example must be reproducible");
    }
}
