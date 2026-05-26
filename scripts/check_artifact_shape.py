#!/usr/bin/env python3
"""
check_artifact_shape.py — Verify artifact structure in manifests.

Usage:
    python3 scripts/check_artifact_shape.py manifests/examples/*.json

Checks:
    - source_artifact has consistent fields (format, path, checksum structure)
    - generated_artifact (if present) has format, path
    - checksum (if present) has sha256 or md5 as hex strings
    - shard_info (if present) has shard_count, shard_paths
    - size_bytes (if present) is non-negative integer
"""

import json
import sys
import re

HEX_PATTERN = re.compile(r"^[a-fA-F0-9]{32,64}$")


def check_artifact(artifact: dict, label: str, path: str) -> list[str]:
    errors = []

    if not isinstance(artifact, dict):
        errors.append(f"{path}: {label} must be an object")
        return errors

    if not artifact.get("format"):
        errors.append(f"{path}: {label}.format is required")

    # Path is required unless it's a generated_artifact with planned/skipped status
    status = artifact.get("status")
    if not artifact.get("path"):
        if label == "generated_artifact" and status in ("planned", "skipped", None):
            pass  # OK, no path yet
        else:
            errors.append(f"{path}: {label}.path is required")

    # checksum structure
    checksum = artifact.get("checksum")
    if checksum is not None:
        if not isinstance(checksum, dict):
            errors.append(f"{path}: {label}.checksum must be an object")
        else:
            for key in ["sha256", "md5"]:
                value = checksum.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        errors.append(
                            f"{path}: {label}.checksum.{key} must be a string"
                        )
                    elif not HEX_PATTERN.match(value):
                        errors.append(
                            f"{path}: {label}.checksum.{key} does not look like a hex hash"
                        )

    # size_bytes
    size = artifact.get("size_bytes")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
        errors.append(
            f"{path}: {label}.size_bytes must be a non-negative integer"
        )

    # shard_info
    shards = artifact.get("shard_info")
    if shards is not None:
        if not isinstance(shards, dict):
            errors.append(f"{path}: {label}.shard_info must be an object")
        else:
            count = shards.get("shard_count")
            if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 1):
                errors.append(
                    f"{path}: {label}.shard_info.shard_count must be a positive integer"
                )
            paths = shards.get("shard_paths")
            if paths is not None:
                if not isinstance(paths, list):
                    errors.append(
                        f"{path}: {label}.shard_info.shard_paths must be an array"
                    )
                elif count is not None and len(paths) != count:
                    errors.append(
                        f"{path}: {label}.shard_info.shard_paths length ({len(paths)}) "
                        f"!= shard_count ({count})"
                    )

    return errors


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

    # source_artifact
    source = manifest.get("source_artifact")
    if source is None:
        errors.append(f"{path}: source_artifact is required")
    else:
        errors.extend(check_artifact(source, "source_artifact", path))

    # generated_artifact (optional)
    generated = manifest.get("generated_artifact")
    if generated is not None:
        errors.extend(check_artifact(generated, "generated_artifact", path))

    # quantization (optional but checked for shape)
    quant = manifest.get("quantization")
    if quant is not None:
        if not isinstance(quant, dict):
            errors.append(f"{path}: quantization must be an object")
        else:
            bits = quant.get("bits")
            if bits is not None and bits not in {1, 2, 3, 4, 6, 8, 16}:
                errors.append(
                    f"{path}: quantization.bits must be one of [1, 2, 3, 4, 6, 8, 16]"
                )
            group_size = quant.get("group_size")
            if group_size is not None and (
                not isinstance(group_size, int) or group_size < 1
            ):
                errors.append(
                    f"{path}: quantization.group_size must be a positive integer"
                )

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 check_artifact_shape.py <manifest1.json> [...]")
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
        print(f"\n✗ Artifact shape check failed: {len(all_errors)} error(s)")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(sys.argv) - 1} manifest(s) have valid artifact shape")


if __name__ == "__main__":
    main()
