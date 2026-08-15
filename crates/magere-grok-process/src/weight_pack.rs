// SPDX-License-Identifier: GPL-3.0-or-later
//! GOZ1 weight pack format — header + tensor table.

use crate::error::Result;
use crate::types::{GOZ1_MAGIC, GOZ1_VERSION};

/// Tensor type markers in the GOZ1 tensor table.
pub const TENSOR_F16: u8 = 0x01;
pub const TENSOR_TERNARY: u8 = 0x02;

/// Header for a GOZ1 packed checkpoint file.
#[derive(Debug, Clone, Copy)]
pub struct PackHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u32,
    pub tensor_table_offset: u64,
}

impl PackHeader {
    pub fn new(tensor_count: u32, tensor_table_offset: u64) -> Self {
        Self {
            magic: *b"GOZ1",
            version: GOZ1_VERSION,
            tensor_count,
            tensor_table_offset,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20);
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.tensor_count.to_le_bytes());
        buf.extend_from_slice(&self.tensor_table_offset.to_le_bytes());
        buf
    }
}

/// Entry in the GOZ1 tensor table.
#[derive(Debug, Clone)]
pub struct PackTensorEntry {
    pub name_len: u16,
    pub name: String,
    pub dtype: u8,
    pub rank: u8,
    pub shape: Vec<u64>,
    pub byte_len: u64,
    pub data_offset: u64,
}

impl PackTensorEntry {
    pub fn entry_size(&self) -> usize {
        2 + self.name.len() + 1 + 1 + self.shape.len() * 8 + 8 + 8
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.entry_size());
        buf.extend_from_slice(&self.name_len.to_le_bytes());
        buf.extend_from_slice(self.name.as_bytes());
        buf.push(self.dtype);
        buf.push(self.rank);
        for &dim in &self.shape {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&self.byte_len.to_le_bytes());
        buf.extend_from_slice(&self.data_offset.to_le_bytes());
        buf
    }
}

/// In-memory representation of a GOZ1 pack during construction.
#[derive(Debug, Default)]
pub struct PackBuilder {
    entries: Vec<PackTensorEntry>,
    data: Vec<u8>,
}

impl PackBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_tensor(&mut self, name: String, dtype: u8, shape: Vec<u64>, data: Vec<u8>) {
        let byte_len = data.len() as u64;
        let data_offset = self.data.len() as u64;
        self.data.extend_from_slice(&data);

        self.entries.push(PackTensorEntry {
            name_len: name.len() as u16,
            name,
            dtype,
            rank: shape.len() as u8,
            shape,
            byte_len,
            data_offset,
        });
    }

    pub fn finalize(self) -> Result<Vec<u8>> {
        let tensor_count = self.entries.len() as u32;
        let header_size = 20usize;
        let tensor_table_offset = header_size as u64;
        let tensor_table_size: u64 = self.entries.iter().map(|e| e.entry_size() as u64).sum();
        let data_offset = tensor_table_offset + tensor_table_size;

        // Adjust data offsets
        let mut entries = self.entries;
        let mut current_data_offset = data_offset;
        for entry in &mut entries {
            entry.data_offset = current_data_offset;
            current_data_offset += entry.byte_len;
        }

        let header = PackHeader::new(tensor_count, tensor_table_offset);
        let mut output =
            Vec::with_capacity(header_size + tensor_table_size as usize + self.data.len());
        output.extend_from_slice(&header.to_bytes());
        for entry in &entries {
            output.extend_from_slice(&entry.to_bytes());
        }
        output.extend_from_slice(&self.data);

        Ok(output)
    }
}

/// Parse a GOZ1 file from bytes and return (header, entries).
pub fn parse_pack(bytes: &[u8]) -> Option<(PackHeader, Vec<PackTensorEntry>)> {
    if bytes.len() < 20 {
        return None;
    }

    let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
    if magic != *GOZ1_MAGIC {
        return None;
    }

    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != GOZ1_VERSION {
        return None;
    }
    let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let tensor_table_offset = u64::from_le_bytes([
        bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
    ]);

    let header = PackHeader {
        magic,
        version,
        tensor_count,
        tensor_table_offset,
    };

    let mut entries = Vec::with_capacity(tensor_count as usize);
    let mut cursor = tensor_table_offset as usize;

    for _ in 0..tensor_count {
        if cursor + 2 > bytes.len() {
            return None;
        }
        let name_len = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
        cursor += 2;

        if cursor + name_len > bytes.len() {
            return None;
        }
        let name = String::from_utf8_lossy(&bytes[cursor..cursor + name_len]).to_string();
        cursor += name_len;

        if cursor + 2 > bytes.len() {
            return None;
        }
        let dtype = bytes[cursor];
        let rank = bytes[cursor + 1];
        cursor += 2;

        if cursor + rank as usize * 8 > bytes.len() {
            return None;
        }
        let mut shape = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            let dim = u64::from_le_bytes([
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
                bytes[cursor + 4],
                bytes[cursor + 5],
                bytes[cursor + 6],
                bytes[cursor + 7],
            ]);
            shape.push(dim);
            cursor += 8;
        }

        if cursor + 16 > bytes.len() {
            return None;
        }
        let byte_len = u64::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]);
        let data_offset = u64::from_le_bytes([
            bytes[cursor + 8],
            bytes[cursor + 9],
            bytes[cursor + 10],
            bytes[cursor + 11],
            bytes[cursor + 12],
            bytes[cursor + 13],
            bytes[cursor + 14],
            bytes[cursor + 15],
        ]);
        cursor += 16;

        entries.push(PackTensorEntry {
            name_len: name_len as u16,
            name,
            dtype,
            rank,
            shape,
            byte_len,
            data_offset,
        });
    }

    Some((header, entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_pack_and_parse() {
        let mut builder = PackBuilder::new();
        builder.add_tensor("test".into(), TENSOR_F16, vec![2, 2], vec![1, 2, 3, 4]);
        let packed = builder.finalize().unwrap();

        let (header, entries) = parse_pack(&packed).unwrap();
        assert_eq!(&header.magic, GOZ1_MAGIC);
        assert_eq!(header.version, GOZ1_VERSION);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "test");
        assert_eq!(entries[0].byte_len, 4);
    }

    #[test]
    fn parse_rejects_wrong_magic() {
        let mut bytes = vec![b'X', b'X', b'X', b'X'];
        bytes.extend_from_slice(&[0u8; 16]);
        assert!(parse_pack(&bytes).is_none());
    }

    #[test]
    fn parse_rejects_short_input() {
        assert!(parse_pack(b"GOZ").is_none());
    }

    #[test]
    fn parse_rejects_unsupported_version() {
        let mut builder = PackBuilder::new();
        builder.add_tensor("test".into(), TENSOR_F16, vec![2, 2], vec![1, 2, 3, 4]);
        let mut packed = builder.finalize().unwrap();
        packed[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert!(parse_pack(&packed).is_none());
    }
}
