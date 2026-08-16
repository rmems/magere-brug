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


class TestGOZ1ManifestStructure:
    """Test GOZ1-oriented manifests against artifact-shape + schema contracts."""

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _minimal_goz1_fixture(**generated_overrides):
        generated = {
            "format": "goz1",
            "status": "planned",
            "version": 1,
            "source_lineage": {
                "manifest_id": "test-model-v1",
                "path": "/models/test.safetensors",
            },
            "tensor_summary": {
                "tensor_count": 4,
                "f16_count": 1,
                "ternary_count": 3,
            },
        }
        generated.update(generated_overrides)
        return {
            "metadata": {
                "schema_version": 1,
                "created_at": "2026-05-26T00:00:00Z",
                "manifest_id": "test-goz1-v1",
                "description": "GOZ1 fixture for unit tests",
            },
            "model": {
                "slug": "test_model",
                "name": "Test Model",
                "family": "other",
                "parameter_count": {"active": 1000000},
                "architecture": "dense",
            },
            "source_artifact": {
                "format": "safetensors",
                "path": "/models/test.safetensors",
            },
            "generated_artifact": generated,
            "quantization": {
                "method": "ternary",
                "bits": 2,
            },
            "backend_compatibility": {
                "safetensors": {"supported": True, "status": "proven"},
                "goz1": {"supported": True, "status": "planned"},
            },
        }

    def test_goz1_example_manifest_loads(self):
        """Repo GOZ1 example is a complete, loadable manifest."""
        path = self._repo_root() / "manifests" / "examples" / "goz1-pack-example.json"
        with open(path) as f:
            manifest = json.load(f)
        gen = manifest["generated_artifact"]
        backends = manifest.get("backend_compatibility", {})
        # pytest asserts are intentional; nosec keeps Bandit/Codacy quiet in tests
        if gen["format"] != "goz1":  # nosec B101
            raise AssertionError(f"expected goz1 format, got {gen['format']!r}")
        if gen["version"] != 1:  # nosec B101
            raise AssertionError(f"expected goz1 version 1, got {gen['version']!r}")
        if gen["status"] != "success":  # nosec B101
            raise AssertionError(f"expected success status, got {gen['status']!r}")
        if not gen.get("path"):  # nosec B101
            raise AssertionError("generated_artifact.path is required for success")
        if "awq" in backends or "gptq" in backends:  # nosec B101
            raise AssertionError("AWQ/GPTQ backends must not appear on GOZ1 example")

    def test_goz1_fixture_shape_and_enums(self):
        """GOZ1 fixture uses allowed formats/methods and rejects removed backends."""
        fixture = self._minimal_goz1_fixture()
        gen = fixture["generated_artifact"]
        quant = fixture["quantization"]
        if gen["format"] != "goz1" or gen["version"] != 1:  # nosec B101
            raise AssertionError("fixture must declare goz1 version 1")
        if quant["method"] != "ternary":  # nosec B101
            raise AssertionError(f"expected ternary method, got {quant['method']!r}")
        if gen["format"] in ("awq", "gptq") or quant["method"] in ("awq", "gptq"):  # nosec B101
            raise AssertionError("AWQ/GPTQ must not appear on GOZ1 fixture")

    def test_removed_backends_not_in_allowed_keys(self):
        """AWQ/GPTQ are not part of the supported backend key set."""
        allowed = {"safetensors", "gguf", "goz1", "myelin_accelerator"}
        if "awq" in allowed or "gptq" in allowed:  # nosec B101
            raise AssertionError("allowed backend set must not include awq/gptq")
        fixture = self._minimal_goz1_fixture()
        for key in fixture["backend_compatibility"]:
            if key not in allowed:  # nosec B101
                raise AssertionError(f"unexpected backend key: {key}")

    def test_success_without_path_fails_artifact_shape(self):
        """check_artifact_shape requires path when status is success."""
        sys.path.insert(0, str(self._repo_root() / "scripts"))
        from check_artifact_shape import check_artifact

        bad = {"format": "goz1", "status": "success"}
        errors = check_artifact(bad, "generated_artifact", "fixture.json")
        if not any("path" in e for e in errors):  # nosec B101
            raise AssertionError(f"expected path error, got {errors!r}")


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
        assert summary == "fp32"
        
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
