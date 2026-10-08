//! Durable writes and GOZ1 re-read verification.

use magere_grok_process::types::{GOZ1_MAGIC, GOZ1_VERSION};
use magere_grok_process::weight_pack::{PackTensorEntry, TENSOR_F16, TENSOR_TERNARY, parse_pack};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Facts about the pack file on disk, derived by reading it back after the write.
#[derive(Debug, Clone, Copy)]
pub(super) struct PackStats {
    pub tensor_count: u32,
    /// FP16-encoded tensors. Preserve-tier tensors share the FP16 on-disk encoding and are
    /// therefore counted here too.
    pub f16_count: u32,
    pub ternary_count: u32,
    pub size_bytes: u64,
}

/// Write `contents` next to `dest` and rename over it only after a full flush.
pub(super) fn write_atomically(dest: &Path, contents: &[u8]) -> Result<(), String> {
    let tmp = sibling_temp(dest)?;
    write_and_sync(&tmp, contents, dest)?;
    std::fs::rename(&tmp, dest).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!(
            "Failed to replace '{}' with the complete write: {}",
            dest.display(),
            e
        )
    })?;
    Ok(())
}

/// Write the pack to a sibling temp file, flush it, then rename over the destination.
///
/// `File::create` on the destination itself would truncate a previous valid pack before
/// the new bytes are fully on disk.
pub(super) fn write_pack_durably(pack_path: &Path, bytes: &[u8]) -> Result<(), String> {
    write_atomically(pack_path, bytes)
}

/// Re-read the pack from disk and prove it is exactly what we meant to write.
///
/// `expected` is the in-memory buffer the caller just wrote. Comparing against it is what
/// makes this a real verification: `parse_pack` only walks the header and the tensor table,
/// and never checks that each entry's `data_offset + byte_len` is inside the file.
pub(super) fn verify_written_pack(pack_path: &Path, expected: &[u8]) -> Result<PackStats, String> {
    let written = std::fs::read(pack_path)
        .map_err(|e| format!("Failed to re-read pack '{}': {}", pack_path.display(), e))?;
    assert_bytes_match(pack_path, &written, expected)?;
    if !written.starts_with(GOZ1_MAGIC) {
        return Err(format!(
            "pack '{}' does not start with the GOZ1 magic",
            pack_path.display()
        ));
    }
    let (header, entries) = parse_pack(&written).ok_or_else(|| {
        format!(
            "pack '{}' did not round-trip through parse_pack: the GOZ1 magic, version {} or \
             the tensor table did not survive the write",
            pack_path.display(),
            GOZ1_VERSION
        )
    })?;
    let (f16_count, ternary_count) = count_known_dtypes(pack_path, &entries)?;
    Ok(PackStats {
        tensor_count: header.tensor_count,
        f16_count,
        ternary_count,
        size_bytes: written.len() as u64,
    })
}

fn sibling_temp(dest: &Path) -> Result<PathBuf, String> {
    let name = dest.file_name().ok_or_else(|| {
        format!(
            "output path '{}' has no file name for an atomic write",
            dest.display()
        )
    })?;
    Ok(dest.with_file_name(format!(".{}.tmp", name.to_string_lossy())))
}

fn write_and_sync(tmp: &Path, contents: &[u8], dest: &Path) -> Result<(), String> {
    let mut file = File::create(tmp).map_err(|e| {
        format!(
            "Failed to create temporary write for '{}': {}",
            dest.display(),
            e
        )
    })?;
    file.write_all(contents).map_err(|e| {
        format!(
            "Failed to write temporary file for '{}': {}",
            dest.display(),
            e
        )
    })?;
    file.sync_all().map_err(|e| {
        format!(
            "Failed to flush temporary file for '{}' to disk: {}",
            dest.display(),
            e
        )
    })
}

fn assert_bytes_match(pack_path: &Path, written: &[u8], expected: &[u8]) -> Result<(), String> {
    if written.len() != expected.len() {
        return Err(format!(
            "pack '{}' is {} bytes on disk but {} bytes were written -- the file is truncated \
             or was modified underneath us; refusing to checksum or register it",
            pack_path.display(),
            written.len(),
            expected.len()
        ));
    }
    if written == expected {
        return Ok(());
    }
    let first_diff = written
        .iter()
        .zip(expected)
        .position(|(a, b)| a != b)
        .unwrap_or(0);
    Err(format!(
        "pack '{}' does not match the bytes that were written (first difference at offset \
         {}); refusing to checksum or register it",
        pack_path.display(),
        first_diff
    ))
}

fn count_known_dtypes(pack_path: &Path, entries: &[PackTensorEntry]) -> Result<(u32, u32), String> {
    let mut f16_count: u32 = 0;
    let mut ternary_count: u32 = 0;
    for entry in entries {
        match entry.dtype {
            TENSOR_F16 => f16_count += 1,
            TENSOR_TERNARY => ternary_count += 1,
            other => {
                return Err(format!(
                    "pack '{}' tensor '{}' has unknown dtype 0x{:02x}",
                    pack_path.display(),
                    entry.name,
                    other
                ));
            }
        }
    }
    Ok((f16_count, ternary_count))
}
