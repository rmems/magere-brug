"""
Safetensors inspection helpers for magere-brug.

Extracts model metadata from safetensors files for manifest generation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class SafetensorsInspector:
    """Inspect safetensors artifacts and extract metadata."""

    def __init__(self, file_path: str):
        """
        Initialize inspector with a safetensors file path.
        
        Args:
            file_path: Path to the safetensors file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from safetensors file header.
        
        Returns:
            Dictionary with dtype_summary, shard_info, etc.
        """
        try:
            # Read safetensors header
            with open(self.file_path, "rb") as f:
                # First 8 bytes are header length
                header_len_bytes = f.read(8)
                if len(header_len_bytes) < 8:
                    raise ValueError("Invalid safetensors file: too short")
                
                header_len = int.from_bytes(header_len_bytes, byteorder="little")
                header_json = f.read(header_len).decode("utf-8")
                header = json.loads(header_json)
            
            # Extract tensor information (filter out __metadata__)
            tensors = {k: v for k, v in header.items() if isinstance(v, dict) and "dtype" in v}
            dtypes = self._extract_dtypes(tensors)
            
            return {
                "dtype_summary": self._summarize_dtypes(dtypes),
                "tensor_count": len(tensors),
                "tensors": list(tensors.keys()),
                "dtypes": dtypes,
                "file_size_bytes": self.file_path.stat().st_size,
            }
        except Exception as e:
            raise ValueError(f"Failed to extract safetensors metadata: {e}")

    def _extract_dtypes(self, header: Dict) -> Dict[str, int]:
        """Extract dtype distribution from header."""
        dtypes: Dict[str, int] = {}
        for tensor_name, tensor_info in header.items():
            if isinstance(tensor_info, dict) and "dtype" in tensor_info:
                dtype = tensor_info["dtype"]
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
        return dtypes

    def _normalize_dtype(self, dtype: str) -> str:
        normalized = dtype.lower()
        mapping = {
            "f16": "f16",
            "fp16": "fp16",
            "bf16": "bf16",
            "f32": "fp32",
            "fp32": "fp32",
            "i8": "int8",
            "int8": "int8",
            "i4": "int4",
            "int4": "int4",
        }
        return mapping.get(normalized, normalized)

    def _summarize_dtypes(self, dtypes: Dict[str, int]) -> str:
        """Summarize dtypes for manifest."""
        if not dtypes:
            return "unknown"
        
        most_common = max(dtypes.items(), key=lambda x: x[1])
        normalized = self._normalize_dtype(most_common[0])
        if len(dtypes) == 1:
            return normalized
        else:
            return "mixed"


class SafetensorsManifestBuilder:
    """Build manifest entries from safetensors inspection."""

    @staticmethod
    def generate_manifest_snippet(
        file_path: str,
        model_slug: str,
        model_family: str,
    ) -> Dict[str, Any]:
        """
        Generate a manifest snippet for a safetensors model.
        
        Args:
            file_path: Path to safetensors file
            model_slug: Model slug identifier
            model_family: Model family
            
        Returns:
            Dictionary ready for manifest insertion
        """
        inspector = SafetensorsInspector(file_path)
        metadata = inspector.get_metadata()
        
        return {
            "model": {
                "slug": model_slug,
                "family": model_family,
                "architecture": "dense",  # Default, update if MoE
            },
            "source_artifact": {
                "format": "safetensors",
                "path": file_path,
                "dtype_summary": metadata["dtype_summary"],
                "size_bytes": metadata["file_size_bytes"],
                "shard_info": {
                    "shard_count": 1,
                    "shard_size_bytes": metadata["file_size_bytes"],
                }
            },
            "metadata": {
                "tensor_count": metadata["tensor_count"],
                "dtypes": metadata["dtypes"],
            }
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: inspect_safetensors.py <path_to_safetensors_file>")
        sys.exit(1)
    
    try:
        inspector = SafetensorsInspector(sys.argv[1])
        metadata = inspector.get_metadata()
        print(json.dumps(metadata, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
