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
    missing_meta = REQUIRED_METADATA - set(metadata.keys())
    if missing_meta:
        errors.append(f"{path}: Missing metadata fields: {missing_meta}")

    if metadata.get("schema_version", 0) < 1:
        errors.append(
            f"{path}: metadata.schema_version must be >= 1"
        )

    if not metadata.get("created_at"):
        errors.append(f"{path}: metadata.created_at is required")

    if not metadata.get("manifest_id"):
        errors.append(f"{path}: metadata.manifest_id is required")

    # model
    model = manifest.get("model", {})
    missing_model = REQUIRED_MODEL - set(model.keys())
    if missing_model:
        errors.append(f"{path}: Missing model fields: {missing_model}")

    if model.get("slug") and not isinstance(model["slug"], str):
        errors.append(f"{path}: model.slug must be a string")

    if model.get("architecture") not in VALID_ARCHITECTURES:
        errors.append(
            f"{path}: model.architecture must be one of {VALID_ARCHITECTURES}"
        )

    param_count = model.get("parameter_count", {})
    active = param_count.get("active")
    if active is not None and (not isinstance(active, int) or active < 0):
        errors.append(
            f"{path}: model.parameter_count.active must be a non-negative integer"
        )

    # source_artifact
    source = manifest.get("source_artifact", {})
    if not source.get("format"):
        errors.append(f"{path}: source_artifact.format is required")
    elif source["format"] not in VALID_SOURCE_FORMATS:
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
