//! Inter-process registry lock and load/store helpers.

use crate::registry::ArtifactRegistry;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

/// Exclusive flock on `{registry}.lock`, held for the whole pack run.
pub(super) struct RegistryLock {
    _file: File,
    path: PathBuf,
}

impl RegistryLock {
    pub(super) fn acquire(registry_file: &Path) -> Result<Self, String> {
        let path = lock_path(registry_file);
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create registry lock directory '{}': {}",
                    parent.display(),
                    e
                )
            })?;
        }
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| format!("Failed to open registry lock '{}': {}", path.display(), e))?;
        lock_exclusive(&file, &path)?;
        Ok(Self { _file: file, path })
    }
}

impl Drop for RegistryLock {
    fn drop(&mut self) {
        let _ = unlock(&self._file);
        let _ = self.path;
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

#[cfg(unix)]
fn lock_exclusive(file: &File, path: &Path) -> Result<(), String> {
    flock(file, libc_lock_ex())
        .map_err(|e| format!("Failed to lock registry '{}': {}", path.display(), e))
}

#[cfg(unix)]
fn unlock(file: &File) -> std::io::Result<()> {
    flock(file, libc_lock_un())
}

#[cfg(unix)]
fn flock(file: &File, operation: i32) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: `file` is an open fd for the duration of the call; flock is a documented
    // POSIX advisory lock on that fd.
    let rc = unsafe { posix_flock(file.as_raw_fd(), operation) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
const fn libc_lock_ex() -> i32 {
    2
}

#[cfg(unix)]
const fn libc_lock_un() -> i32 {
    8
}

#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "flock"]
    fn posix_flock(fd: i32, operation: i32) -> i32;
}

#[cfg(not(unix))]
fn lock_exclusive(_file: &File, path: &Path) -> Result<(), String> {
    Err(format!(
        "registry locking is only implemented on Unix; cannot lock '{}'",
        path.display()
    ))
}

#[cfg(not(unix))]
fn unlock(_file: &File) -> std::io::Result<()> {
    Ok(())
}
