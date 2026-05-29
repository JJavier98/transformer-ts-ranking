"""Helpers for reading and writing benchmark manifests and artifacts.

The benchmark persists intermediate manifests and reports to disk at each stage.
These helpers centralize YAML/JSON serialization so callers do not duplicate
directory creation, encoding choices or payload validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_directory(path: Path) -> Path:
    """Create a directory path when needed and return it for chaining.

    Args:
        path: Directory to create, including any missing parent directories.

    Returns:
        The same ``Path`` instance received by the function.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and validate that it contains a mapping payload.

    Args:
        path: YAML file to read.

    Returns:
        Parsed YAML contents as a dictionary.

    Raises:
        TypeError: If the YAML root object is not a mapping.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict payload in {path}, got {type(payload).__name__}")
    return payload


def write_yaml(payload: dict[str, Any], path: Path) -> Path:
    """Persist a mapping payload as YAML.

    Args:
        payload: Mapping to serialize.
        path: Output YAML path.

    Returns:
        The output path after writing the file.
    """
    ensure_directory(path.parent)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def write_json(payload: dict[str, Any], path: Path) -> Path:
    """Persist a mapping payload as indented JSON.

    Args:
        payload: Mapping to serialize.
        path: Output JSON path.

    Returns:
        The output path after writing the file.
    """
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
