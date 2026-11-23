//! Shared registry lock and atomic persistence.
//!
//! `magere register`, `magere recipe apply`, and `magere pack-goz1` all
//! read-modify-write the same registry file. They take this exclusive lock for
//! the whole update so one command cannot replace a snapshot that missed the
//! other's entry.

use crate::atomic_write::write_bytes_atomically;
use crate::registry::ArtifactRegistry;
use fs4::FileExt;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

/// Exclusive lock on `{registry}.lock`, held for a registry read-modify-write.
pub struct RegistryLock {
    file: File,
}

impl RegistryLock {
    pub fn acquire(registry_file: &Path) -> Result<Self, String> {
        let path = lock_path(&registry_identity(registry_file));
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
fn registry_identity(registry_file: &Path) -> PathBuf {
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

fn lock_path(registry_file: &Path) -> PathBuf {
    let mut path = registry_file.as_os_str().to_os_string();
    path.push(".lock");
    PathBuf::from(path)
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
