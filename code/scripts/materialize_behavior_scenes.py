#!/usr/bin/env python3
"""Reconstruct all UniETP BEHAVIOR scenes from official BEHAVIOR assets.

Each benchmark entry contains an exact recursive JSON patch against the pinned
official ``*_best.json`` scene. Whitespace and object-key order may differ,
but the reconstructed JSON value is verified against a canonical SHA-256.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import string
import sys
import tempfile
from typing import Any, Mapping


PATCH_FORMAT = "unietp-behavior-exact-patch-v1"
BEHAVIOR_ASSETS_VERSION = "3.7.2rc1"
RECIPE_KEYS = {
    "base_scene_file",
    "base_sha256",
    "target_canonical_sha256",
    "patch",
}


class MaterializationError(RuntimeError):
    """Raised when a benchmark recipe cannot be materialized safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MaterializationError(f"Cannot canonicalize reconstructed JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"Cannot read JSON file {path}: {exc}") from exc


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise MaterializationError(f"{label} must be a JSON object.")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise MaterializationError(f"{label} must be a SHA-256 hex digest.")
    return value.lower()


def _resolve_under(root: Path, relative: str, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise MaterializationError(f"{label} must be relative, got {relative!r}.")
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise MaterializationError(
            f"{label} escapes its root directory: {relative!r}."
        ) from exc
    return resolved


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                data,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            handle.write("\n")
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _apply_recursive_patch(base: Any, patch: Any, label: str) -> Any:
    patch = _require_mapping(patch, label)
    if set(patch) == {"v"}:
        return copy.deepcopy(patch["v"])
    if set(patch) != {"r", "c"}:
        raise MaterializationError(
            f"{label} must be either a replacement {{'v': ...}} or "
            "a dictionary patch {'r': [...], 'c': {...}}."
        )
    if not isinstance(base, dict):
        raise MaterializationError(f"{label} expects a JSON object in the base scene.")

    remove = patch["r"]
    if not isinstance(remove, list) or not all(isinstance(key, str) for key in remove):
        raise MaterializationError(f"{label}.remove must be a list of strings.")
    if len(remove) != len(set(remove)):
        raise MaterializationError(f"{label}.remove contains duplicate keys.")
    changes = _require_mapping(patch["c"], f"{label}.c")
    overlap = set(remove) & set(changes)
    if overlap:
        raise MaterializationError(
            f"{label} removes and changes the same keys: {sorted(overlap)!r}."
        )

    # Copy only dictionaries on changed paths. Unchanged JSON subtrees are
    # immutable during materialization and can be shared with the cached base.
    result = dict(base)
    for key in remove:
        if key not in result:
            raise MaterializationError(f"{label} removes missing key {key!r}.")
        del result[key]
    for key, child_patch in changes.items():
        child_label = f"{label}.c[{key!r}]"
        if key in result:
            result[key] = _apply_recursive_patch(result[key], child_patch, child_label)
        else:
            child_patch = _require_mapping(child_patch, child_label)
            if set(child_patch) != {"v"}:
                raise MaterializationError(
                    f"{child_label} adds a key and must use a replacement patch."
                )
            result[key] = _apply_recursive_patch(None, child_patch, child_label)
    return result


def _find_key(value: Any, forbidden_key: str, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == forbidden_key:
                return child_path
            match = _find_key(child, forbidden_key, child_path)
            if match is not None:
                return match
    elif isinstance(value, list):
        for index, child in enumerate(value):
            match = _find_key(child, forbidden_key, f"{path}[{index}]")
            if match is not None:
                return match
    return None


def _materialize_recipe(
    recipe: Mapping[str, Any],
    assets_root: Path,
    label: str,
    base_cache: dict[Path, tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    if set(recipe) != RECIPE_KEYS:
        raise MaterializationError(
            f"{label} does not match {PATCH_FORMAT!r}; expected keys "
            f"{sorted(RECIPE_KEYS)!r}, got {sorted(recipe)!r}."
        )

    base_relative = recipe.get("base_scene_file")
    if not isinstance(base_relative, str) or not base_relative:
        raise MaterializationError(f"{label}.base_scene_file must be a string.")
    base_path = _resolve_under(assets_root, base_relative, "base_scene_file")
    if not base_path.is_file():
        raise MaterializationError(
            f"Official BEHAVIOR base scene is missing: {base_path}"
        )
    expected_base_hash = _require_sha256(
        recipe.get("base_sha256"), f"{label}.base_sha256"
    )

    if base_path not in base_cache:
        actual_base_hash = _sha256(base_path)
        base = _require_mapping(_load_json(base_path), str(base_path))
        base_cache[base_path] = (actual_base_hash, base)
    actual_base_hash, base = base_cache[base_path]
    if actual_base_hash.lower() != expected_base_hash:
        raise MaterializationError(
            f"Official base scene hash mismatch for {base_path}. "
            f"Expected {expected_base_hash}, got {actual_base_hash}. "
            f"Install BEHAVIOR assets version {BEHAVIOR_ASSETS_VERSION}."
        )

    scene = _apply_recursive_patch(base, recipe.get("patch"), f"{label}.patch")
    if not isinstance(scene, dict):
        raise MaterializationError(f"{label} reconstructed a non-object JSON root.")

    versions = _require_mapping(scene.get("versions"), f"{label}.versions")
    behavior_assets = _require_mapping(
        versions.get("behavior-1k-assets"),
        f"{label}.versions['behavior-1k-assets']",
    )
    if "version" in behavior_assets:
        raise MaterializationError(
            f"{label} patch must not provide behavior-1k-assets.version; "
            "the materializer owns this global value."
        )
    behavior_assets = dict(behavior_assets)
    behavior_assets["version"] = BEHAVIOR_ASSETS_VERSION
    versions = dict(versions)
    versions["behavior-1k-assets"] = behavior_assets
    scene = dict(scene)
    scene["versions"] = versions

    forbidden_path = _find_key(scene, "git_hash")
    if forbidden_path is not None:
        raise MaterializationError(
            f"{label} reconstructed forbidden provenance field {forbidden_path}."
        )
    expected_target_hash = _require_sha256(
        recipe.get("target_canonical_sha256"),
        f"{label}.target_canonical_sha256",
    )
    actual_target_hash = _canonical_sha256(scene)
    if actual_target_hash != expected_target_hash:
        raise MaterializationError(
            f"Reconstructed scene hash mismatch for {label}. "
            f"Expected {expected_target_hash}, got {actual_target_hash}."
        )
    return scene


def materialize_all(
    benchmark_path: Path,
    assets_root: Path,
    output_dir: Path | None,
    force: bool,
) -> int:
    benchmark = _load_json(benchmark_path)
    if not isinstance(benchmark, list):
        raise MaterializationError("BEHAVIOR benchmark root must be a JSON list.")

    benchmark_root = benchmark_path.resolve().parent
    base_cache: dict[Path, tuple[str, Mapping[str, Any]]] = {}
    seen_outputs: set[Path] = set()
    pending_writes: list[tuple[Path, dict[str, Any]]] = []
    for task_index, task in enumerate(benchmark):
        task = _require_mapping(task, f"task {task_index}")
        sampled_scenes = task.get("sampled_scenes")
        if not isinstance(sampled_scenes, list) or not sampled_scenes:
            raise MaterializationError(f"task {task_index} has no sampled_scenes.")
        for scene_index, sampled_scene in enumerate(sampled_scenes):
            sampled_scene = _require_mapping(
                sampled_scene, f"task {task_index} scene {scene_index}"
            )
            scene_args = _require_mapping(
                sampled_scene.get("args"), f"task {task_index} scene {scene_index} args"
            )
            scene_file = scene_args.get("scene_file")
            if not isinstance(scene_file, str) or not scene_file:
                raise MaterializationError(
                    f"task {task_index} scene {scene_index} has no scene_file."
                )
            recipe = _require_mapping(
                scene_args.get("scene_recipe"),
                f"task {task_index} scene {scene_index} scene_recipe",
            )
            if output_dir is None:
                destination = _resolve_under(benchmark_root, scene_file, "scene_file")
            else:
                destination = _resolve_under(
                    output_dir.resolve(), Path(scene_file).name, "scene_file name"
                )
            if destination in seen_outputs:
                raise MaterializationError(
                    f"Multiple benchmark entries write the same scene: {destination}"
                )
            seen_outputs.add(destination)
            if destination.exists() and not force:
                raise MaterializationError(
                    f"Output already exists: {destination}. "
                    "Remove it or rerun with --force."
                )

            scene = _materialize_recipe(
                recipe,
                assets_root,
                f"task {task_index} scene {scene_index}",
                base_cache,
            )
            pending_writes.append((destination, scene))

    if not pending_writes:
        raise MaterializationError("The benchmark did not contain any scene recipes.")
    for destination, scene in pending_writes:
        _atomic_write_json(destination, scene)
    return len(pending_writes)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct every UniETP BEHAVIOR scene JSON exactly, ignoring "
            "whitespace and object-key order, from official BEHAVIOR scenes "
            "and patches embedded in BEHAVIOR.json."
        )
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        type=Path,
        help="Path to the downloaded UniETP BEHAVIOR.json.",
    )
    parser.add_argument(
        "--behavior-assets-root",
        required=True,
        type=Path,
        help="Official behavior-1k-assets directory containing scenes/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional output directory. By default, each scene_file path is "
            "resolved relative to BEHAVIOR.json."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace scene JSON files that already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        count = materialize_all(
            benchmark_path=args.benchmark.resolve(),
            assets_root=args.behavior_assets_root.resolve(),
            output_dir=args.output_dir,
            force=args.force,
        )
    except MaterializationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Materialized {count} exact BEHAVIOR scene JSON files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
