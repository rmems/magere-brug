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
        let path = lock_path(registry_file);
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

pub fn load_registry(registry_file: &Path) -> Result<ArtifactRegistry, String> {
    if !registry_file.exists() {
        return Ok(ArtifactRegistry::new());
    }
    let content = std::fs::read_to_string(registry_file).map_err(|e| {
        format!(
            "Failed to read registry '{}': {}",
            registry_file.display(),
            e
        )
    })?;
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
    write_bytes_atomically(registry_file, registry_json.as_bytes())
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
}
