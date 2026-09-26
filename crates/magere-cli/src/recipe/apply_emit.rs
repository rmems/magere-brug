//! Register apply: output tree creation, path containment, and symlink-safe writes.

use super::{Recipe, resolve::is_filename_safe_id};
use std::path::{Path, PathBuf};

pub(super) fn emit_base_dir(recipe: &Recipe, registry_path: &Path) -> PathBuf {
    recipe
        .outputs
        .as_ref()
        .and_then(|outputs| outputs.output_dir.as_deref())
        .map(PathBuf::from)
        .or_else(|| {
            registry_path.parent().and_then(|parent| {
                if parent.as_os_str().is_empty() {
                    None
                } else {
                    Some(parent.to_path_buf())
                }
            })
        })
        .unwrap_or_else(|| PathBuf::from("."))
}

pub(super) fn prepare_emit_tree(base: &Path) -> Result<PathBuf, String> {
    reject_symlink_components(base, "output base")?;
    std::fs::create_dir_all(base).map_err(|e| {
        format!(
            "failed to create output base directory {}: {e}",
            base.display()
        )
    })?;
    reject_symlink_components(base, "output base")?;
    let canonical_base = base
        .canonicalize()
        .map_err(|e| format!("failed to resolve output base {}: {e}", base.display()))?;
    for subdir in ["manifests", "handoff"] {
        ensure_emit_subdirectory(&canonical_base, subdir)?;
    }
    Ok(canonical_base)
}

fn ensure_emit_subdirectory(canonical_base: &Path, subdir: &str) -> Result<(), String> {
    let dir = canonical_base.join(subdir);
    reject_symlink_components(&dir, subdir)?;
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create {subdir} directory {}: {e}", dir.display()))?;
    reject_symlink_components(&dir, subdir)
}

fn reject_symlink_components(path: &Path, label: &str) -> Result<(), String> {
    let mut prefix = PathBuf::new();
    for component in path.components() {
        prefix.push(component);
        if prefix.as_os_str().is_empty() {
            continue;
        }
        let metadata = match std::fs::symlink_metadata(&prefix) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => {
                return Err(format!(
                    "failed to inspect {label} path component {}: {error}",
                    prefix.display()
                ));
            }
        };
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "refusing symlink in {label} path at {}",
                prefix.display()
            ));
        }
    }
    Ok(())
}

pub(super) fn write_regular_file_under_base(
    output_base: &Path,
    path: &Path,
    contents: &[u8],
    label: &str,
) -> Result<(), String> {
    reject_symlink_components(path, label)?;
    ensure_path_under_base(output_base, path, label)?;
    write_regular_file(path, contents)
}

fn ensure_path_under_base(output_base: &Path, path: &Path, label: &str) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Err(format!(
            "{label} path '{}' must include a parent directory",
            path.display()
        ));
    };
    let parent = parent.canonicalize().map_err(|e| {
        format!(
            "failed to resolve {label} directory '{}': {e}",
            parent.display()
        )
    })?;
    let base = output_base.canonicalize().map_err(|e| {
        format!(
            "failed to resolve output base '{}': {e}",
            output_base.display()
        )
    })?;
    if parent == base || parent.starts_with(&base) {
        Ok(())
    } else {
        Err(format!(
            "{label} path '{}' escapes the selected output base '{}'",
            path.display(),
            base.display()
        ))
    }
}

fn write_regular_file(path: &Path, contents: &[u8]) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(format!(
                "refusing to overwrite symlink at {}",
                path.display()
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!("failed to inspect {}: {error}", path.display()));
        }
    }
    std::fs::write(path, contents).map_err(|e| format!("failed to write {}: {e}", path.display()))
}

pub(super) fn reject_registration_path_collisions(
    registry_path: &Path,
    source_manifest: &Path,
    emitted_manifest: &Path,
    emitted_handoff: &Path,
) -> Result<(), String> {
    let paths = [
        ("registry", registry_path),
        ("source manifest", source_manifest),
        ("emitted manifest", emitted_manifest),
        ("combine-for-AI handoff", emitted_handoff),
    ];
    let mut canonical: Vec<(String, PathBuf)> = Vec::new();
    for (label, path) in paths {
        let resolved = canonicalize_registration_path(path, label)?;
        for (other_label, other) in &canonical {
            if *other == resolved {
                if paths_alias_in_place_emit(label, other_label) {
                    continue;
                }
                return Err(format!(
                    "registration paths must not alias the same file: {label} '{}' and {other_label} '{}'",
                    path.display(),
                    other.display()
                ));
            }
        }
        canonical.push((label.to_string(), resolved));
    }
    Ok(())
}

fn paths_alias_in_place_emit(left: &str, right: &str) -> bool {
    matches!(
        (left, right),
        ("source manifest", "emitted manifest") | ("emitted manifest", "source manifest")
    )
}

fn canonicalize_registration_path(path: &Path, label: &str) -> Result<PathBuf, String> {
    if path.exists() {
        return path
            .canonicalize()
            .map_err(|e| format!("failed to resolve {label} path '{}': {e}", path.display()));
    }
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("{label} path '{}' must name a file", path.display()))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = parent.canonicalize().map_err(|e| {
        format!(
            "failed to resolve {label} directory '{}': {e}",
            parent.display()
        )
    })?;
    Ok(parent.join(file_name))
}

pub(super) fn emit_file_name(manifest_id: &str) -> Result<String, String> {
    if is_filename_safe_id(manifest_id) {
        Ok(format!("{manifest_id}.json"))
    } else {
        Err(format!(
            "metadata.manifest_id '{manifest_id}' must be a single filename-safe path component"
        ))
    }
}

pub(super) fn write_emitted_manifest(
    output_base: &Path,
    source_path: &Path,
    bytes: &[u8],
    dest: &Path,
) -> Result<(), String> {
    if same_existing_file(source_path, dest) {
        return Ok(());
    }
    write_regular_file_under_base(output_base, dest, bytes, "emitted manifest")
        .map_err(|e| format!("failed to emit artifact manifest {}: {e}", dest.display()))
}

fn same_existing_file(left: &Path, right: &Path) -> bool {
    right.exists() && left.canonicalize().ok() == right.canonicalize().ok()
}
