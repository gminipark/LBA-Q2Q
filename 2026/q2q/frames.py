"""DramaQA video frame discovery and representative frame sampling."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from q2q.config import DEFAULT_DATASET_ROOT, IMAGE_SUFFIXES, SCRIPT_DIR
from q2q.schemas import SceneRecord


def resolve_image_root(requested: Path | None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend(
        [
            DEFAULT_DATASET_ROOT / "images",
            SCRIPT_DIR / "images",
            SCRIPT_DIR / "AnotherMissOh_images" / "dramaqa_frames",
            SCRIPT_DIR / "AnotherMissOh_images",
        ]
    )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        nested = candidate / "dramaqa_frames"
        if nested.is_dir():
            return nested
        if candidate.is_dir():
            return candidate
    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find the image root. Searched:\n  - {searched}")



def natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)



def evenly_spaced(items: Sequence[Path], count: int) -> list[Path]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    unique_indices = list(dict.fromkeys(indices))
    return [items[index] for index in unique_indices]



def representative_frames(
    image_root: Path,
    records: Sequence[SceneRecord],
    max_images: int,
    shot_ranges: dict[str, tuple[int, int]] | None = None,
) -> list[tuple[str, Path]]:
    """Select one middle frame per shot, spread fairly across scenes."""

    shot_ranges = shot_ranges or {}
    groups: list[tuple[str, list[Path]]] = []
    for record in records:
        parts = record.scene_id.split("_")
        if len(parts) < 2:
            continue
        episode_dir, scene_dir = parts[0], parts[1]
        directory = image_root / episode_dir / scene_dir
        if not directory.is_dir():
            continue

        candidates: list[Path] = []
        for shot_dir in sorted(
            (path for path in directory.iterdir() if path.is_dir()), key=natural_sort_key
        ):
            if record.scene_id in shot_ranges:
                numbers = re.findall(r"\d+", shot_dir.name)
                if numbers:
                    shot_number = int(numbers[-1])
                    start, end = shot_ranges[record.scene_id]
                    if not start <= shot_number <= end:
                        continue
            frames = sorted(
                (
                    path
                    for path in shot_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ),
                key=natural_sort_key,
            )
            if frames:
                candidates.append(frames[len(frames) // 2])
        if candidates:
            groups.append((record.scene_id, candidates))

    if not groups or max_images <= 0:
        return []

    allocations = [0] * len(groups)
    remaining = max_images
    active = list(range(len(groups)))
    while remaining > 0 and active:
        next_active: list[int] = []
        for index in active:
            if remaining <= 0:
                break
            capacity = len(groups[index][1])
            if allocations[index] < capacity:
                allocations[index] += 1
                remaining -= 1
            if allocations[index] < capacity:
                next_active.append(index)
        active = next_active

    selected: list[tuple[str, Path]] = []
    for (scene_id, candidates), count in zip(groups, allocations):
        for path in evenly_spaced(candidates, count):
            selected.append((scene_id, path))
    return selected

