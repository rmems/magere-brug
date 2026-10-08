use super::harness::*;
use crate::pack::io::verify_written_pack;
use crate::pack::*;
use magere_grok_process::weight_pack::parse_pack;
use std::fs;
use tempfile::TempDir;

fn valid_pack_bytes() -> Vec<u8> {
    let h = default_harness();
    let outcome = run_pack_recipe(&h.recipe_path, Some(&h.registry_path), None).unwrap();
    fs::read(&outcome.pack_path).unwrap()
}

#[test]
fn verify_rejects_a_tail_truncated_pack() {
    let bytes = valid_pack_bytes();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("truncated.goz1");
    let truncated = &bytes[..bytes.len() - (FIXTURE_TENSORS as usize * 4)];
    fs::write(&path, truncated).unwrap();
    let (header, entries) = parse_pack(truncated).expect("truncated pack still parses");
    assert_eq!(header.tensor_count, FIXTURE_TENSORS);
    assert_eq!(entries.len() as u32, FIXTURE_TENSORS);
    let err = verify_written_pack(&path, &bytes).unwrap_err();
    assert!(err.contains("truncated"), "{}", err);
}

#[test]
fn verify_rejects_a_corrupted_byte() {
    let bytes = valid_pack_bytes();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("corrupt.goz1");
    let mut corrupt = bytes.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0xff;
    fs::write(&path, &corrupt).unwrap();
    let err = verify_written_pack(&path, &bytes).unwrap_err();
    assert!(err.contains("does not match"), "{}", err);
    assert!(err.contains(&last.to_string()), "{}", err);
}

#[test]
fn verify_rejects_a_pack_without_the_goz1_magic() {
    let bytes = valid_pack_bytes();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("nomagic.goz1");
    let mut clobbered = bytes.clone();
    clobbered[..4].copy_from_slice(b"NOPE");
    fs::write(&path, &clobbered).unwrap();
    let err = verify_written_pack(&path, &clobbered).unwrap_err();
    assert!(err.contains("GOZ1 magic"), "{}", err);
}

#[test]
fn verify_rejects_a_pack_with_the_wrong_version() {
    let bytes = valid_pack_bytes();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("badversion.goz1");
    let mut clobbered = bytes.clone();
    clobbered[4..8].copy_from_slice(&99u32.to_le_bytes());
    fs::write(&path, &clobbered).unwrap();
    let err = verify_written_pack(&path, &clobbered).unwrap_err();
    assert!(err.contains("did not round-trip"), "{}", err);
}

#[test]
fn verify_rejects_a_pack_with_an_unknown_dtype() {
    let bytes = valid_pack_bytes();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("baddtype.goz1");
    let mut clobbered = bytes.clone();
    let table_offset = u64::from_le_bytes(clobbered[12..20].try_into().unwrap()) as usize;
    let name_len = u16::from_le_bytes(
        clobbered[table_offset..table_offset + 2]
            .try_into()
            .unwrap(),
    ) as usize;
    clobbered[table_offset + 2 + name_len] = 0x7f;
    fs::write(&path, &clobbered).unwrap();
    let err = verify_written_pack(&path, &clobbered).unwrap_err();
    assert!(err.contains("unknown dtype"), "{}", err);
}

#[test]
fn verify_accepts_the_pack_the_command_actually_wrote() {
    let bytes = valid_pack_bytes();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("good.goz1");
    fs::write(&path, &bytes).unwrap();
    let stats = verify_written_pack(&path, &bytes).expect("valid pack verifies");
    assert_eq!(stats.tensor_count, FIXTURE_TENSORS);
    assert_eq!(stats.ternary_count, FIXTURE_TERNARY);
    assert_eq!(stats.f16_count, FIXTURE_F16);
    assert_eq!(stats.size_bytes, bytes.len() as u64);
}
