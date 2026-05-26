#!/usr/bin/env python3
"""
check_required_fields.py — Verify required fields in model manifests.

Usage:
    python3 scripts/check_required_fields.py manifests/examples/*.json

Checks:
    - metadata.schema_version >= 1
    - metadata.created_at is present
    - metadata.manifest_id is present
    - model.slug, model.name, model.family are present
    - model.parameter_count.active >= 0
    - model.architecture is 'dense' or 'moe'
    - source_artifact.format is valid
    - source_artifact.path is present
"""

import json
import sys

REQUIRED_METADATA = {"schema_version", "created_at", "manifest_id"}
REQUIRED_MODEL = {"slug", "name", "family", "parameter_count", "architecture"}
VALID_SOURCE_FORMATS = {"safetensors", "gguf", "hf_repo", "local_dir"}
VALID_ARCHITECTURES = {"dense", "moe"}


def check_manifest(path: str) -> list[str]:
    errors = []
    try:
        with open(path, "r") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in {path}: {e}")
        return errors
    except FileNotFoundError:
        errors.append(f"File not found: {path}")
        return errors

    # metadata
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        errors.append(f"{path}: metadata must be an object")
        metadata = {}
    missing_meta = REQUIRED_METADATA - set(metadata.keys())
    if missing_meta:
        errors.append(f"{path}: Missing metadata fields: {missing_meta}")

    schema_version = metadata.get("schema_version", 0)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
        errors.append(
            f"{path}: metadata.schema_version must be >= 1"
        )

    if not metadata.get("created_at"):
        errors.append(f"{path}: metadata.created_at is required")
    elif not isinstance(metadata["created_at"], str):
        errors.append(f"{path}: metadata.created_at must be a string")

    if not metadata.get("manifest_id"):
        errors.append(f"{path}: metadata.manifest_id is required")
    elif not isinstance(metadata["manifest_id"], str):
        errors.append(f"{path}: metadata.manifest_id must be a string")

    # model
    model = manifest.get("model", {})
    if not isinstance(model, dict):
        errors.append(f"{path}: model must be an object")
        model = {}
    missing_model = REQUIRED_MODEL - set(model.keys())
    if missing_model:
        errors.append(f"{path}: Missing model fields: {missing_model}")

    if "slug" in model and not isinstance(model["slug"], str):
        errors.append(f"{path}: model.slug must be a string")

    if "name" in model and not isinstance(model["name"], str):
        errors.append(f"{path}: model.name must be a string")

    if "family" in model and not isinstance(model["family"], str):
        errors.append(f"{path}: model.family must be a string")

    architecture = model.get("architecture")
    if not isinstance(architecture, str) or architecture not in VALID_ARCHITECTURES:
        errors.append(
            f"{path}: model.architecture must be one of {VALID_ARCHITECTURES}"
        )

    param_count = model.get("parameter_count", {})
    if not isinstance(param_count, dict):
        errors.append(f"{path}: model.parameter_count must be an object")
    elif "active" not in param_count:
        errors.append(f"{path}: model.parameter_count.active is required")
    else:
        active = param_count["active"]
        if not isinstance(active, int) or isinstance(active, bool) or active < 0:
            errors.append(
                f"{path}: model.parameter_count.active must be a non-negative integer"
            )

    # source_artifact
    source = manifest.get("source_artifact", {})
    if not isinstance(source, dict):
        errors.append(f"{path}: source_artifact must be an object")
        source = {}
    if not source.get("format"):
        errors.append(f"{path}: source_artifact.format is required")
    elif not isinstance(source["format"], str) or source["format"] not in VALID_SOURCE_FORMATS:
        errors.append(
            f"{path}: source_artifact.format must be one of {VALID_SOURCE_FORMATS}"
        )

    if not source.get("path"):
        errors.append(f"{path}: source_artifact.path is required")

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 check_required_fields.py <manifest1.json> [...]")
        sys.exit(1)

    all_errors = []
    for path in sys.argv[1:]:
        errors = check_manifest(path)
        all_errors.extend(errors)
        if not errors:
            print(f"✓ {path}")
        else:
            for err in errors:
                print(f"✗ {err}")

    if all_errors:
        print(f"\n✗ Required fields check failed: {len(all_errors)} error(s)")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(sys.argv) - 1} manifest(s) have required fields")


if __name__ == "__main__":
    main()
