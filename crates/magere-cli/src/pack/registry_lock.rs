//! Inter-process registry lock and load/store helpers.

use crate::registry::ArtifactRegistry;
use fs4::FileExt;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

/// Exclusive lock on `{registry}.lock`, held for the whole pack run.
pub(super) struct RegistryLock {
    file: File,
}

impl RegistryLock {
    pub(super) fn acquire(registry_file: &Path) -> Result<Self, String> {
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

pub(super) fn load_registry(registry_file: &Path) -> Result<ArtifactRegistry, String> {
    if !registry_file.exists() {
        return Ok(ArtifactRegistry::new());
    }
    let content = std::fs::read_to_string(registry_file)
        .map_err(|e| format!("Failed to read registry: {}", e))?;
    ArtifactRegistry::from_json(&content).map_err(|e| format!("Failed to parse registry: {}", e))
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
