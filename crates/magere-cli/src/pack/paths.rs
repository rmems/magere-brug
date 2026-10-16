//! Output layout, path collision checks, and source-manifest resolution.

use crate::manifest::Manifest;
use crate::recipe::{Recipe, is_filename_safe_id};
use crate::registry::{ArtifactRegistry, RegistryEntry};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Derived pack and manifest destinations for one run.
#[derive(Debug, Clone)]
pub(super) struct OutputLayout {
    pub output_dir: PathBuf,
    pub manifest_id: String,
    pub pack_path: PathBuf,
    pub manifest_path: PathBuf,
}

pub(super) fn resolve_output_layout(
    recipe: &Recipe,
    source: &Manifest,
    output_dir_override: Option<&Path>,
) -> Result<OutputLayout, String> {
    let output_dir: PathBuf = match output_dir_override {
        Some(dir) => dir.to_path_buf(),
        None => recipe
            .outputs
            .as_ref()
            .and_then(|o| o.output_dir.as_deref())
            .map(PathBuf::from)
            .ok_or_else(|| {
                format!(
                    "recipe '{}' has no outputs.output_dir; pass --output-dir",
                    recipe.recipe_id
                )
            })?,
    };

    let manifest_id = recipe
        .outputs
        .as_ref()
        .and_then(|o| o.manifest_id.clone())
        .unwrap_or_else(|| format!("{}-goz1", source.metadata.manifest_id));
    let manifest_id = filename_stem(&manifest_id)?.to_string();
    let pack_path = output_dir.join(format!("{}.goz1", manifest_id));
    let manifest_path = output_dir.join(format!("{}.manifest.json", manifest_id));
    Ok(OutputLayout {
        output_dir,
        manifest_id,
        pack_path,
        manifest_path,
    })
}

/// Load the source manifest from a filesystem path or a registry slug / manifest_id.
pub(super) fn resolve_source_manifest(
    recipe: &Recipe,
    registry: &ArtifactRegistry,
) -> Result<(PathBuf, Manifest), String> {
    let raw = recipe
        .inputs
        .as_ref()
        .and_then(|i| i.source_manifest.as_deref())
        .ok_or_else(|| {
            format!(
                "recipe '{}' is missing inputs.source_manifest",
                recipe.recipe_id
            )
        })?;
    let path = resolve_source_ref(raw, registry)?;
    let source = Manifest::from_file(&path)
        .map_err(|e| format!("Failed to load source manifest '{}': {}", path.display(), e))?;
    source
        .validate()
        .map_err(|e| format!("source manifest '{}' is invalid: {}", path.display(), e))?;
    Ok((path, source))
}

fn resolve_source_ref(raw: &str, registry: &ArtifactRegistry) -> Result<PathBuf, String> {
    let as_path = Path::new(raw);
    if as_path.exists() {
        return Ok(as_path.to_path_buf());
    }
    if looks_like_filesystem_path(raw) {
        return Err(format!(
            "Failed to load source manifest '{}': file not found",
            raw
        ));
    }
    let entry = registry
        .lookup(raw)
        .or_else(|| registry.lookup_by_manifest_id(raw))
        .ok_or_else(|| {
            format!(
                "inputs.source_manifest '{}' is not a readable path and is not a slug or \
                 manifest_id in the selected registry",
                raw
            )
        })?;
    match &entry.manifest_path {
        Some(path) if path.is_file() => Ok(path.clone()),
        Some(path) => Err(format!(
            "registry id '{}' points at missing manifest '{}'",
            raw,
            path.display()
        )),
        None => Err(format!(
            "registry id '{}' is registered but has no stored manifest_path; re-register \
             the source with `magere register`",
            raw
        )),
    }
}

fn looks_like_filesystem_path(raw: &str) -> bool {
    raw.contains('/') || raw.contains('\\') || raw.ends_with(".json")
}

/// One filename component, using the same rule as other recipe output ids.
fn filename_stem(manifest_id: &str) -> Result<&str, String> {
    let trimmed = manifest_id.trim();
    if is_filename_safe_id(trimmed) {
        return Ok(trimmed);
    }
    Err(format!(
        "outputs.manifest_id '{manifest_id}' must be a single filename with no path separators or '..' segments"
    ))
}

/// Refuse when a derived output path would overwrite a recipe input or the registry file.
pub(super) fn reject_aliased_outputs(
    recipe_path: &Path,
    source_manifest_path: &Path,
    dissect_manifest_path: &Path,
    registry_path: &Path,
    pack_path: &Path,
    manifest_path: &Path,
) -> Result<(), String> {
    let mut protected = Vec::with_capacity(4);
    for input in [recipe_path, source_manifest_path, dissect_manifest_path] {
        protected.push(canonicalize_existing(input, "input")?);
    }
    protected.push(resolve_planned(registry_path)?);
    for output in [pack_path, manifest_path] {
        let resolved = resolve_planned(output)?;
        if let Some(hit) = protected.iter().find(|path| paths_alias(path, &resolved)) {
            return Err(format!(
                "refusing to run: output '{}' would overwrite recipe input '{}'",
                output.display(),
                hit.display()
            ));
        }
    }
    Ok(())
}

fn canonicalize_existing(path: &Path, kind: &str) -> Result<PathBuf, String> {
    std::fs::canonicalize(path)
        .map_err(|e| format!("Failed to resolve {kind} '{}': {}", path.display(), e))
}

pub(super) fn resolve_planned_path(path: &Path) -> Result<PathBuf, String> {
    resolve_planned(path)
}

fn resolve_planned(path: &Path) -> Result<PathBuf, String> {
    if path.exists() {
        return canonicalize_existing(path, "path");
    }
    let (parent, file_name) = split_planned(path)?;
    resolve_under_parent(path, parent, file_name)
}

fn split_planned(path: &Path) -> Result<(&Path, &OsStr), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("output path '{}' has no parent", path.display()))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("output path '{}' has no file name", path.display()))?;
    Ok((parent, file_name))
}

fn resolve_under_parent(path: &Path, parent: &Path, file_name: &OsStr) -> Result<PathBuf, String> {
    if parent.as_os_str().is_empty() {
        let cwd = std::env::current_dir()
            .map_err(|e| format!("Failed to resolve current directory: {e}"))?;
        return Ok(cwd.join(file_name));
    }
    if !parent.exists() {
        return Ok(path.to_path_buf());
    }
    let canonical_parent = canonicalize_existing(parent, "output dir")?;
    Ok(canonical_parent.join(file_name))
}

fn paths_alias(left: &Path, right: &Path) -> bool {
    left == right || same_file::is_same_file(left, right).unwrap_or(false)
}

/// Refuse when a derived output path would overwrite another registry entry's files.
///
/// Slug and manifest-id ownership live on [`ArtifactRegistry::admit_pack`].
pub(super) fn reject_output_aliases(
    registry: &ArtifactRegistry,
    slug: &str,
    pack_path: &Path,
    manifest_path: &Path,
) -> Result<(), String> {
    for entry in registry.models.values() {
        if entry.slug == slug {
            continue;
        }
        if entry_aliases_outputs(entry, pack_path, manifest_path)? {
            return Err(format!(
                "output path '{}' collides with registry slug '{}' (manifest '{}'). \
                 Nothing was written.",
                pack_path.display(),
                entry.slug,
                entry
                    .manifest_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| entry.manifest_id.clone())
            ));
        }
    }
    Ok(())
}

fn entry_aliases_outputs(
    entry: &RegistryEntry,
    pack_path: &Path,
    manifest_path: &Path,
) -> Result<bool, String> {
    let Some(stored) = &entry.manifest_path else {
        return Ok(false);
    };
    let ours = [resolve_planned(pack_path)?, resolve_planned(manifest_path)?];
    let mut theirs = vec![resolve_planned(stored)?];
    if let Some(pack) = sibling_goz1(stored) {
        theirs.push(resolve_planned(&pack)?);
    }
    Ok(ours
        .iter()
        .any(|output| theirs.iter().any(|existing| paths_alias(output, existing))))
}

fn sibling_goz1(manifest: &Path) -> Option<PathBuf> {
    let name = manifest.file_name()?.to_string_lossy();
    let stem = name.strip_suffix(".manifest.json")?;
    Some(manifest.with_file_name(format!("{stem}.goz1")))
}
