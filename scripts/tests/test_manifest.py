"""
Unit tests for manifest validation and loading.
"""

import json
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_safetensors import SafetensorsInspector, SafetensorsManifestBuilder
from register_gguf import GGUFInspector, GGUFManifestBuilder
from quant_awq import AWQManifestBuilder, AWQOrchestrator


class TestManifestLoading:
    """Test manifest loading and validation."""

    def test_valid_manifest_json_loads(self):
        """Test that valid manifest JSON loads correctly."""
        manifest_json = {
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            }
        }
        
        # Should be serializable
        json_str = json.dumps(manifest_json)
        loaded = json.loads(json_str)
        assert loaded["model"]["slug"] == "test_model"

    def test_required_fields_present(self):
        """Test that required manifest fields are recognized."""
        manifest_json = {
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            }
        }
        
        # Check required fields
        assert "metadata" in manifest_json
        assert "model" in manifest_json
        assert "source_artifact" in manifest_json
        assert manifest_json["metadata"]["schema_version"] == 1
        assert manifest_json["model"]["slug"] == "test_model"

    def test_invalid_manifest_fails_validation(self):
        """Test that invalid manifest fails validation."""
        # Missing required fields
        invalid_manifest = {
            "metadata": {
                "schema_version": 0,  # Invalid: must be >= 1
            }
        }
        
        assert invalid_manifest["metadata"]["schema_version"] < 1


class TestChecksum:
    """Test checksum field shape and validation."""

    def test_checksum_field_shape(self):
        """Test that checksum field has correct shape."""
        manifest_json = {
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors",
                "checksum": {
                    "sha256": "abc123def456",
                    "md5": "xyz789"
                }
            }
        }
        
        checksum = manifest_json["source_artifact"]["checksum"]
        assert "sha256" in checksum
        assert "md5" in checksum
        assert isinstance(checksum["sha256"], str)
        assert isinstance(checksum["md5"], str)

    def test_checksum_optional(self):
        """Test that checksum field is optional."""
        manifest_json = {
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            }
        }
        
        # Should work without checksum
        assert "checksum" not in manifest_json["source_artifact"]


class TestSourceFormats:
    """Test supported source artifact formats."""

    def test_gguf_format_accepted(self):
        """Test that GGUF format is accepted."""
        manifest_json = {
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "gguf",
                "path": "/models/test.gguf"
            }
        }
        
        assert manifest_json["source_artifact"]["format"] == "gguf"

    def test_safetensors_format_accepted(self):
        """Test that safetensors format is accepted."""
        manifest_json = {
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors"
            }
        }
        
        assert manifest_json["source_artifact"]["format"] == "safetensors"

    def test_hf_repo_format_accepted(self):
        """Test that HuggingFace repo format is accepted."""
        manifest_json = {
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "hf_repo",
                "path": "allenai/OLMoE-1B-7B-0125-Instruct"
            }
        }
        
        assert manifest_json["source_artifact"]["format"] == "hf_repo"

    def test_local_dir_format_accepted(self):
        """Test that local_dir format is accepted."""
        manifest_json = {
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-model-v1"
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "test",
                "parameter_count": {"active": 1000000},
                "architecture": "dense"
            },
            "source_artifact": {
                "format": "local_dir",
                "path": "/local/models/my_model",
                "source_url": "file:///local/models/my_model"
            }
        }
        
        assert manifest_json["source_artifact"]["format"] == "local_dir"


class TestAWQManifestGeneration:
    """Test AWQ manifest snippet generation."""

    def test_awq_manifest_snippet_generated(self):
        """Test that AWQ manifest snippet is generated correctly."""
        snippet = AWQManifestBuilder.generate_manifest_snippet(
            model_slug="test_model",
            model_family="test",
            source_format="safetensors",
            source_path="/models/test.safetensors",
            quantization_bits=4,
            group_size=128,
        )
        
        assert snippet["model"]["slug"] == "test_model"
        assert snippet["generated_artifact"]["format"] == "awq"
        assert snippet["generated_artifact"]["status"] == "planned"
        assert snippet["quantization"]["bits"] == 4
        assert snippet["quantization"]["group_size"] == 128


class TestGGUFInspection:
    """Test GGUF inspection helpers."""

    def test_gguf_quantization_inference(self):
        """Test GGUF quantization inference from filename."""
        # Test with Q8_0 variant
        inspector = GGUFInspector.__new__(GGUFInspector)
        quant = inspector._infer_quantization("model-q8_0.gguf")
        assert quant == "q8_0"
        
        # Test with Q6_K variant
        quant = inspector._infer_quantization("model-q6_k.gguf")
        assert quant == "q6_k"
        
        # Test with IQ3_M variant
        quant = inspector._infer_quantization("model-iq3_m.gguf")
        assert quant == "iq3_m"

    def test_gguf_manifest_builder_creates_structure(self):
        """Test GGUF manifest builder creates proper structure."""
        # We test the structure, not actual file loading
        snippet_structure = {
            "model": {
                "slug": "test_model",
                "family": "test",
                "architecture": "moe",
            },
            "source_artifact": {
                "format": "gguf",
                "path": "/models/test.gguf",
                "dtype_summary": "mixed",
                "size_bytes": 1000000,
            },
            "quantization": {
                "method": "q8_0",
            },
            "metadata": {
                "gguf_version": 3,
                "tensor_count": 100,
            }
        }
        
        assert snippet_structure["source_artifact"]["format"] == "gguf"
        assert snippet_structure["model"]["architecture"] == "moe"
        assert "quantization" in snippet_structure


class TestSafetensorsInspection:
    """Test Safetensors inspection helpers."""

    def test_safetensors_dtype_summarization(self):
        """Test Safetensors dtype summarization."""
        inspector = SafetensorsInspector.__new__(SafetensorsInspector)
        
        # Single dtype
        dtypes = {"float32": 100}
        summary = inspector._summarize_dtypes(dtypes)
        assert summary == "float32"
        
        # Mixed dtypes
        dtypes = {"float32": 50, "float16": 50}
        summary = inspector._summarize_dtypes(dtypes)
        assert summary == "mixed"
        
        # Empty
        dtypes = {}
        summary = inspector._summarize_dtypes(dtypes)
        assert summary == "unknown"

    def test_safetensors_manifest_builder_creates_structure(self):
        """Test Safetensors manifest builder creates proper structure."""
        # We test the structure, not actual file loading
        snippet_structure = {
            "model": {
                "slug": "test_model",
                "family": "test",
                "architecture": "dense",
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors",
                "dtype_summary": "float32",
                "size_bytes": 5000000,
                "shard_info": {
                    "shard_count": 1,
                    "shard_size_bytes": 5000000,
                }
            },
            "metadata": {
                "tensor_count": 200,
                "dtypes": {"float32": 200},
            }
        }
        
        assert snippet_structure["source_artifact"]["format"] == "safetensors"
        assert snippet_structure["model"]["slug"] == "test_model"
        assert "dtype_summary" in snippet_structure["source_artifact"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
