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
    let prior_mode = existing_mode(dest);
    temporary.persist(dest).map_err(|e| {
        format!(
            "Failed to replace '{}' with the complete write: {}",
            dest.display(),
            e.error
        )
    })?;
    apply_output_permissions(dest, prior_mode);
    Ok(())
}

/// Permission bits `dest` had before the write, when it existed.
#[cfg(unix)]
fn existing_mode(dest: &Path) -> Option<u32> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(dest)
        .ok()
        .map(|metadata| metadata.permissions().mode() & 0o777)
}

#[cfg(not(unix))]
fn existing_mode(_dest: &Path) -> Option<u32> {
    None
}

/// `NamedTempFile` creates its file mode 0600 on Unix and `persist` keeps that
/// mode across the rename. Restore shareable permissions: the previous mode
/// when the file existed, else 0644 — the mode a umask-respecting write would
/// have produced — so packs, manifests, and registries stay readable by other
/// users (e.g. containers running under a different uid).
#[cfg(unix)]
fn apply_output_permissions(dest: &Path, prior_mode: Option<u32>) {
    use std::os::unix::fs::PermissionsExt;
    let mode = prior_mode.unwrap_or(0o644);
    let _ = std::fs::set_permissions(dest, std::fs::Permissions::from_mode(mode));
}

#[cfg(not(unix))]
fn apply_output_permissions(_dest: &Path, _prior_mode: Option<u32>) {}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    #[test]
    fn written_files_are_group_and_world_readable() {
        use super::write_bytes_atomically;
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("fresh.bin");
        write_bytes_atomically(&path, b"data").unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o644,
            "a fresh file must not keep the temporary file's 0600 mode"
        );

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        write_bytes_atomically(&path, b"again").unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640,
            "an existing file keeps its own permission bits across a rewrite"
        );
    }
}
