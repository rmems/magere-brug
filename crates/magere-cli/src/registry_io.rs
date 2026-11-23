//! Shared registry lock and atomic persistence.
//!
//! `magere register`, `magere recipe apply`, and `magere pack-goz1` all
//! read-modify-write the same registry file. They take this exclusive lock for
//! the whole update so one command cannot replace a snapshot that missed the
//! other's entry.

use crate::atomic_write::write_bytes_atomically;
use crate::registry::ArtifactRegistry;
use fs4::FileExt;
use std::ffi::OsStr;
use std::fs::{File, OpenOptions};
use std::path::{Component, Path, PathBuf};

/// Exclusive lock on `{registry}.lock`, held for a registry read-modify-write.
pub struct RegistryLock {
    file: File,
}

impl RegistryLock {
    pub fn acquire(registry_file: &Path) -> Result<Self, String> {
        let identity = registry_identity(registry_file);
        // Both spellings are checked: a raw `*.lock` name is reserved, and an
        // alias whose target ends in `.lock` would collide just the same.
        reject_lock_named_registry(registry_file)?;
        reject_lock_named_registry(&identity)?;
        let path = lock_path(&identity);
        ensure_lock_parent(&path)?;
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| format!("Failed to open registry lock '{}': {}", path.display(), e))?;
        FileExt::lock(&file)
            .map_err(|e| format!("Failed to lock registry '{}': {}", path.display(), e))?;
        Ok(Self { file })
    }
}

impl Drop for RegistryLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

/// A registry named `*.lock` shares a file with the lock of the registry whose
/// name drops that suffix — locking one would create it over the other's
/// registry bytes. Refuse the name outright instead.
fn reject_lock_named_registry(registry_file: &Path) -> Result<(), String> {
    if registry_file
        .file_name()
        .is_some_and(|name| name.to_string_lossy().ends_with(".lock"))
    {
        return Err(format!(
            "registry path '{}' ends in '.lock' and would collide with the lock \
             files of other registries; choose a different file name",
            registry_file.display()
        ));
    }
    Ok(())
}

/// Load the artifact registry from disk.
///
/// The sibling `*.lock` file coordinates cooperating `magere` processes only; it
/// does not stop a foreign writer from replacing the registry while this call reads.
pub fn load_registry(registry_file: &Path) -> Result<ArtifactRegistry, String> {
    let content = match std::fs::read_to_string(registry_file) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ArtifactRegistry::new()),
        Err(e) => {
            return Err(format!(
                "Failed to read registry '{}': {}",
                registry_file.display(),
                e
            ));
        }
    };
    ArtifactRegistry::from_json(&content).map_err(|e| {
        format!(
            "Failed to parse registry '{}': {}",
            registry_file.display(),
            e
        )
    })
}

pub fn store_registry(registry: &ArtifactRegistry, registry_file: &Path) -> Result<(), String> {
    let registry_json = registry
        .to_json_pretty()
        .map_err(|e| format!("Failed to serialize registry: {}", e))?;
    write_bytes_atomically(&registry_identity(registry_file), registry_json.as_bytes())
}

/// The file the registry path actually names.
///
/// Aliases — a final-component symlink, a symlinked parent directory, or a
/// `a/../a` spelling — all converge on one canonical path so they share a
/// single lock file, and so an atomic write updates the real file instead of
/// replacing the symlink that pointed at it.
///
/// Every caller that locks, loads, collision-checks against, or stores the
/// registry must use this same destination: a dangling symlink resolves to
/// its target here even though `Path::exists` reports the link as missing.
pub(crate) fn registry_identity(registry_file: &Path) -> PathBuf {
    let resolved = follow_final_symlink(registry_file);
    if let Ok(canonical) = std::fs::canonicalize(&resolved) {
        return canonical;
    }
    // The file does not exist yet: canonicalize its parent so lexical and
    // symlinked-directory aliases still converge on the same lock name.
    canonicalize_parent(&resolved).unwrap_or_else(|| registry_file.to_path_buf())
}

/// Follow a final-component symlink (and short chains) so a write through an
/// alias lands on the real file instead of replacing the link.
fn follow_final_symlink(path: &Path) -> PathBuf {
    let mut resolved = path.to_path_buf();
    for _ in 0..8 {
        match symlink_target(&resolved) {
            Some(target) => resolved = target,
            None => return resolved,
        }
    }
    resolved
}

fn symlink_target(path: &Path) -> Option<PathBuf> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    if !meta.file_type().is_symlink() {
        return None;
    }
    let target = std::fs::read_link(path).ok()?;
    Some(if target.is_absolute() {
        target
    } else {
        path.parent().unwrap_or_else(|| Path::new(".")).join(target)
    })
}

fn canonicalize_parent(path: &Path) -> Option<PathBuf> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())?;
    let name = path.file_name()?;
    std::fs::canonicalize(parent)
        .ok()
        .map(|parent| parent.join(name))
}

/// Resolve `path` as the directory entry it names, tolerating ancestors that
/// do not exist yet: the deepest existing ancestor is canonicalized and the
/// missing tail plus the literal file name is re-appended. A final-component
/// symlink is never followed — the link itself is the entry — and directory
/// aliases are frozen at call time, so a directory symlink retargeted while a
/// lock is held cannot move the entry out from under the resolved registry
/// destination.
pub(crate) fn entry_identity(path: &Path) -> Result<PathBuf, String> {
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("path '{}' must name a file", path.display()))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let (ancestor, suffix) = nearest_existing_ancestor(parent, file_name);
    let resolved = if ancestor.as_os_str().is_empty() {
        std::env::current_dir()
            .map_err(|e| format!("Failed to resolve current directory: {e}"))?
            .join(parent)
            .join(file_name)
    } else {
        ancestor
            .canonicalize()
            .map_err(|e| {
                format!(
                    "failed to resolve path directory '{}': {e}",
                    ancestor.display()
                )
            })?
            .join(suffix)
    };
    Ok(normalize_lexically(&resolved))
}

/// Walk `parent` up to the nearest existing ancestor, accumulating the missing
/// tail (file name included) into `suffix`. An empty `ancestor` means no
/// component exists. Missing `..` components are kept in `suffix` — they
/// still matter once the missing directories exist.
fn nearest_existing_ancestor(parent: &Path, file_name: &OsStr) -> (PathBuf, PathBuf) {
    let mut suffix = PathBuf::from(file_name);
    let mut cursor = parent;
    while !cursor.as_os_str().is_empty() && !cursor.exists() {
        match cursor.file_name() {
            Some(name) => suffix = PathBuf::from(name).join(&suffix),
            None if cursor.ends_with("..") => suffix = PathBuf::from("..").join(&suffix),
            None => {}
        }
        cursor = cursor.parent().unwrap_or(Path::new(""));
    }
    (cursor.to_path_buf(), suffix)
}

/// Fold `.`/`..` lexically so a planned path whose missing tail escapes an
/// ancestor compares equal to the path the writer will actually create. `..`
/// that climbs past the root is dropped, as the filesystem would; `..` under
/// a relative root is preserved.
fn normalize_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir if normalized.file_name().is_some() => {
                normalized.pop();
            }
            Component::ParentDir if !normalized.has_root() => normalized.push(".."),
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn lock_path(registry_file: &Path) -> PathBuf {
    let mut path = registry_file.as_os_str().to_os_string();
    path.push(".lock");
    PathBuf::from(path)
}

/// The lock file `RegistryLock::acquire` would take for `registry_file`,
/// resolved through the same identity rules — for callers that must prove two
/// paths do not share a lock before acquiring either.
pub(crate) fn registry_lock_path(registry_file: &Path) -> PathBuf {
    lock_path(&registry_identity(registry_file))
}

fn ensure_lock_parent(path: &Path) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() || parent.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(parent).map_err(|e| {
        format!(
            "Failed to create registry lock directory '{}': {}",
            parent.display(),
            e
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn a_second_lock_waits_until_the_first_is_dropped() {
        let dir = tempfile::TempDir::new().unwrap();
        let registry = dir.path().join("registry.json");
        let held = RegistryLock::acquire(&registry).unwrap();
        let (started, started_rx) = mpsc::channel();
        let (acquired, acquired_rx) = mpsc::channel();
        thread::spawn(move || {
            started.send(()).unwrap();
            let _lock = RegistryLock::acquire(&registry).unwrap();
            acquired.send(()).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            acquired_rx
                .recv_timeout(Duration::from_millis(200))
                .is_err(),
            "the second lock must wait while the first is held"
        );
        drop(held);
        acquired_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    }

    #[test]
    fn a_registry_named_lock_is_refused() {
        let dir = tempfile::TempDir::new().unwrap();
        let err = RegistryLock::acquire(&dir.path().join("registry.lock"))
            .err()
            .expect("a '.lock' registry name must be refused");
        assert!(err.contains("'.lock'"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_alias_to_a_lock_named_target_is_refused() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("models.lock");
        std::fs::write(&target, b"{}").unwrap();
        let alias = dir.path().join("legacy.json");
        symlink(&target, &alias).unwrap();
        let err = RegistryLock::acquire(&alias)
            .err()
            .expect("an alias whose target ends in '.lock' must be refused");
        assert!(err.contains("'.lock'"), "{err}");
    }

    #[test]
    fn entry_identity_tolerates_missing_ancestors_and_folds_dotdot() {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let spelled = root
            .join("missing")
            .join("..")
            .join("planned")
            .join("registry.json");
        let entry = entry_identity(&spelled).unwrap();
        assert_eq!(entry, root.join("planned").join("registry.json"));
    }

    #[cfg(unix)]
    #[test]
    fn entry_identity_freezes_a_directory_alias() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let v1 = root.join("v1");
        let v2 = root.join("v2");
        std::fs::create_dir_all(&v1).unwrap();
        std::fs::create_dir_all(&v2).unwrap();
        let alias = root.join("current");
        symlink(&v1, &alias).unwrap();

        let spelled = alias.join("registry.json");
        let pinned = entry_identity(&spelled).unwrap();
        assert_eq!(pinned, v1.join("registry.json"));

        // Retarget the directory alias: the pinned entry still names v1, so a
        // caller that resolved once keeps lock, load, store, and emission on
        // the same release.
        std::fs::remove_file(&alias).unwrap();
        symlink(&v2, &alias).unwrap();
        assert_eq!(registry_identity(&pinned), v1.join("registry.json"));
        assert_ne!(entry_identity(&spelled).unwrap(), pinned);
    }

    #[test]
    fn missing_registry_file_loads_as_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("no-such-registry.json");
        let registry = load_registry(&path).unwrap();
        assert_eq!(registry.count(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_registry_writes_through_to_its_target() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("real-registry.json");
        std::fs::write(&target, "{\"version\":1,\"models\":{}}").unwrap();
        let alias = dir.path().join("alias-registry.json");
        symlink(&target, &alias).unwrap();

        let mut registry = ArtifactRegistry::new();
        registry.version = 7;
        store_registry(&registry, &alias).unwrap();

        assert!(
            std::fs::symlink_metadata(&alias)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the store must update the target, not replace the symlink"
        );
        let stored: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&target).unwrap()).unwrap();
        assert_eq!(stored["version"], serde_json::json!(7));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_registry_aliases_share_one_lock() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("real-registry.json");
        std::fs::write(&target, "{\"version\":1,\"models\":{}}").unwrap();
        let alias = dir.path().join("alias-registry.json");
        symlink(&target, &alias).unwrap();

        let held = RegistryLock::acquire(&target).unwrap();
        let (started, started_rx) = mpsc::channel();
        let (acquired, acquired_rx) = mpsc::channel();
        thread::spawn(move || {
            started.send(()).unwrap();
            let _lock = RegistryLock::acquire(&alias).unwrap();
            acquired.send(()).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            acquired_rx
                .recv_timeout(Duration::from_millis(200))
                .is_err(),
            "a lock through a symlink alias must wait on the target's lock"
        );
        drop(held);
        acquired_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    }
}
