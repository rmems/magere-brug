//! Unique-temporary-file writes shared by the registry and the pack runner.

use std::io::Write;
use std::path::Path;

/// Write `contents` to a uniquely named temporary file in `dest`'s directory,
/// flush it, then rename over `dest`.
///
/// The temporary name comes from [`tempfile::NamedTempFile`], created with
/// exclusive create, so an existing recipe input is never truncated just
/// because it happens to match a predictable `*.tmp` path.
pub fn write_bytes_atomically(dest: &Path, contents: &[u8]) -> Result<(), String> {
    let parent = dest
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if parent != Path::new(".") {
        std::fs::create_dir_all(parent).map_err(|e| {
            format!(
                "Failed to create directory '{}' for '{}': {}",
                parent.display(),
                dest.display(),
                e
            )
        })?;
    }
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|e| {
        format!(
            "Failed to create temporary write for '{}': {}",
            dest.display(),
            e
        )
    })?;
    temporary.write_all(contents).map_err(|e| {
        format!(
            "Failed to write temporary file for '{}': {}",
            dest.display(),
            e
        )
    })?;
    temporary.as_file().sync_all().map_err(|e| {
        format!(
            "Failed to flush temporary file for '{}' to disk: {}",
            dest.display(),
            e
        )
    })?;
    temporary.persist(dest).map_err(|e| {
        format!(
            "Failed to replace '{}' with the complete write: {}",
            dest.display(),
            e.error
        )
    })?;
    Ok(())
}
