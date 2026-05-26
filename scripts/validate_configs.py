#!/usr/bin/env python3
"""
validate_configs.py — Validate model lineup configs and cloud stubs.

Usage:
    python3 scripts/validate_configs.py configs/models/*.json
    python3 scripts/validate_configs.py configs/models/cloud/*.json

Checks:
    - JSON is valid
    - Required top-level fields exist
    - Models array has required fields per entry
    - Cloud stubs have security flags (stub, enabled, requires_secrets)
"""

import json
import sys
from pathlib import Path

REQUIRED_TOP_LEVEL = {"schema_version", "created_at", "models"}
REQUIRED_MODEL_FIELDS = {"slug", "family", "architecture"}
CLOUD_STUB_REQUIRED = {"stub", "status", "enabled", "requires_secrets"}
VALID_ARCHITECTURES = {"dense", "moe"}


def validate_config(path: str) -> list[str]:
    errors = []
    try:
        with open(path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in {path}: {e}")
        return errors
    except FileNotFoundError:
        errors.append(f"File not found: {path}")
        return errors

    # Top-level fields
    if not isinstance(config, dict):
        errors.append(f"{path}: Root value must be an object")
        return errors
    
    missing = REQUIRED_TOP_LEVEL - set(config.keys())
    if missing:
        errors.append(f"{path}: Missing top-level fields: {missing}")

    schema_version = config.get("schema_version")
    if schema_version is not None and (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version < 1
    ):
        errors.append(f"{path}: schema_version must be an integer >= 1")

    created_at = config.get("created_at")
    if created_at is not None and not isinstance(created_at, str):
        errors.append(f"{path}: created_at must be a string")

    # Models array
    models = config.get("models", [])
    if not isinstance(models, list):
        errors.append(f"{path}: 'models' must be an array")
        return errors

    for idx, model in enumerate(models):
        if not isinstance(model, dict):
            errors.append(f"{path}: models[{idx}] is not an object")
            continue

        missing_model = REQUIRED_MODEL_FIELDS - set(model.keys())
        if missing_model:
            errors.append(
                f"{path}: models[{idx}] missing fields: {missing_model}"
            )

        for field in REQUIRED_MODEL_FIELDS:
            if field in model and not isinstance(model[field], str):
                errors.append(
                    f"{path}: models[{idx}].{field} must be a string"
                )

        architecture = model.get("architecture")
        if isinstance(architecture, str) and architecture not in VALID_ARCHITECTURES:
            errors.append(
                f"{path}: models[{idx}].architecture must be one of {VALID_ARCHITECTURES}"
            )

    # Cloud stub security check
    if "cloud" in path.lower():
        missing_stub = CLOUD_STUB_REQUIRED - set(config.keys())
        if missing_stub:
            errors.append(
                f"{path}: Cloud stub missing security fields: {missing_stub}"
            )
        else:
            if config.get("enabled") is True:
                errors.append(
                    f"{path}: Cloud stub should not be enabled (enabled=true)"
                )
            if config.get("stub") is not True:
                errors.append(
                    f"{path}: Cloud stub should have stub=true"
                )
            # Local providers don't need secrets; cloud providers do
            provider = config.get("provider", "")
            is_local = isinstance(provider, str) and "local" in provider.lower()
            if not is_local and config.get("requires_secrets") is not True:
                errors.append(
                    f"{path}: Cloud stub should require secrets (requires_secrets=true)"
                )

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 validate_configs.py <config1.json> [config2.json ...]")
        sys.exit(1)

    all_errors = []
    for path in sys.argv[1:]:
        errors = validate_config(path)
        all_errors.extend(errors)
        if not errors:
            print(f"✓ {path}")
        else:
            for err in errors:
                print(f"✗ {err}")

    if all_errors:
        print(f"\n✗ Validation failed: {len(all_errors)} error(s)")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(sys.argv) - 1} config(s) validated successfully")


if __name__ == "__main__":
    main()
