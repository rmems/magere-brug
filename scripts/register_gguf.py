"""
GGUF registration helpers for magere-brug.

Extracts model metadata from GGUF files for manifest generation.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any


class GGUFInspector:
    """Inspect GGUF artifacts and extract metadata."""

    def __init__(self, file_path: str):
        """
        Initialize inspector with a GGUF file path.
        
        Args:
            file_path: Path to the GGUF file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from GGUF file.
        
        Returns:
            Dictionary with dtype_summary, quantization info, etc.
        """
        try:
            # Read GGUF header (simplified extraction)
            with open(self.file_path, "rb") as f:
                # GGUF files start with magic bytes
                magic = f.read(4)
                if magic != b"GGUF":
                    raise ValueError("Invalid GGUF file: wrong magic bytes")
                
                version = int.from_bytes(f.read(4), byteorder="little")
                tensor_count = int.from_bytes(f.read(8), byteorder="little")
                metadata_count = int.from_bytes(f.read(8), byteorder="little")
            
            return {
                "format": "gguf",
                "version": version,
                "tensor_count": tensor_count,
                "metadata_count": metadata_count,
                "dtype_summary": "mixed",  # GGUF typically mixed
                "file_size_bytes": self.file_path.stat().st_size,
                "quantization_format": self._infer_quantization(self.file_path.name),
            }
        except Exception as e:
            raise ValueError(f"Failed to extract GGUF metadata: {e}")

    @staticmethod
    def _infer_quantization(filename: str) -> str:
        """Infer quantization format from filename."""
        filename_lower = filename.lower()
        
        if "q8_0" in filename_lower:
            return "q8_0"
        elif "q6_k" in filename_lower:
            return "q6_k"
        elif "q5_k" in filename_lower:
            return "q5_k"
        elif "q4" in filename_lower:
            return "q4"
        elif "f16" in filename_lower or "fp16" in filename_lower:
            return "f16"
        elif "iq" in filename_lower:
            # IQ quantization variants
            for variant in ["iq4_nl", "iq3_m", "iq2_xxs"]:
                if variant in filename_lower:
                    return variant
        
        return "unknown"


class GGUFManifestBuilder:
    """Build manifest entries from GGUF inspection."""

    @staticmethod
    def generate_manifest_snippet(
        file_path: str,
        model_slug: str,
        model_family: str,
        architecture: str = "dense",
    ) -> Dict[str, Any]:
        """
        Generate a manifest snippet for a GGUF model.
        
        Args:
            file_path: Path to GGUF file
            model_slug: Model slug identifier
            model_family: Model family
            architecture: Model architecture (dense or moe)
            
        Returns:
            Dictionary ready for manifest insertion
        """
        inspector = GGUFInspector(file_path)
        metadata = inspector.get_metadata()
        
        return {
            "model": {
                "slug": model_slug,
                "family": model_family,
                "architecture": architecture,
            },
            "source_artifact": {
                "format": "gguf",
                "path": file_path,
                "dtype_summary": metadata["dtype_summary"],
                "size_bytes": metadata["file_size_bytes"],
            },
            "quantization": {
                "method": metadata["quantization_format"],
            },
            "metadata": {
                "gguf_version": metadata["version"],
                "tensor_count": metadata["tensor_count"],
            }
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: register_gguf.py <path_to_gguf_file>")
        sys.exit(1)
    
    try:
        inspector = GGUFInspector(sys.argv[1])
        metadata = inspector.get_metadata()
        print(json.dumps(metadata, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
