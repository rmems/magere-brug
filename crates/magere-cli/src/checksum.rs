use sha2::{Sha256, Digest};
use std::fs;

/// Compute SHA256 checksum of a file
pub fn compute_file_sha256(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    
    use std::io::Read;
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = [0; 8192];
    
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute SHA256 checksum of a string
pub fn compute_string_sha256(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Verify file checksum against expected SHA256
pub fn verify_checksum(path: &str, expected_sha256: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let actual = compute_file_sha256(path)?;
    Ok(actual.eq_ignore_ascii_case(expected_sha256))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_compute_string_sha256() {
        let hash = compute_string_sha256("test content");
        assert_eq!(hash, "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72");
    }

    #[test]
    fn test_compute_file_sha256() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test content").unwrap();
        
        let hash = compute_file_sha256(temp_file.path().to_str().unwrap()).unwrap();
        assert_eq!(hash, "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72");
    }

    #[test]
    fn test_verify_checksum_valid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test content").unwrap();
        
        let valid = verify_checksum(
            temp_file.path().to_str().unwrap(),
            "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"
        ).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_verify_checksum_invalid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test content").unwrap();
        
        let valid = verify_checksum(
            temp_file.path().to_str().unwrap(),
            "invalid_checksum_hash"
        ).unwrap();
        assert!(!valid);
    }
}
