#!/usr/bin/env python3
"""Interactive DramaQA sub-questioning with Qwen multimodal models.

The script retrieves relevant DramaQA scenes from the local QA/KG files, selects
representative frames, and asks a sequence of strictly binary (yes/no)
sub-questions.  The user's answers are fed back into the conversation until the
model has enough information to answer the original question.

Examples:
    python run.py --show-sample
    python run.py --sample --dry-run --show-context
    python run.py --sample
    python run.py --question "Who does Dokyung meet at the company?"
    python run.py --question "..." --vid AnotherMissOh16_005_0000
"""

from __future__ import annotations

import argparse
import csv
import os
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = (
    SCRIPT_DIR / "DramaQA-20260727T060423Z-1-001" / "DramaQA"
)
DEFAULT_KG_PATH = (
    DEFAULT_DATASET_ROOT / "KG" / "AnotherMissOhQA_test_episode_scene_kg.json"
)
DEFAULT_QA_DIR = DEFAULT_DATASET_ROOT / "QA"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_TYPE_SUBTYPE_MAP_PATH = DEFAULT_OUTPUT_DIR / "type_subtype_map.json"
DEFAULT_TRANSCRIPT_CSV_PATH = DEFAULT_OUTPUT_DIR / "input_example.csv"
DEFAULT_EXPANDED_KG_PATH = DEFAULT_OUTPUT_DIR / "expanded_kg.json"
DEFAULT_EXPANSION_LOG_PATH = DEFAULT_OUTPUT_DIR / "kg_expansion_log.json"
DEFAULT_VISUALIZATION_PATH = (
    DEFAULT_OUTPUT_DIR / "visualization" / "expanded_kg.html"
)

SAMPLE_QUESTION = (
    "Who does Dokyung run into when he goes to Haeyoung1's company with a "
    "bouquet of flowers?"
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ENGLISH_WH_PREFIXES = (
    "what ",
    "where ",
    "when ",
    "who ",
    "whom ",
    "whose ",
    "why ",
    "how ",
    "which ",
)
ENGLISH_BINARY_PREFIXES = (
    "am ",
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "has ",
    "have ",
    "had ",
    "can ",
    "could ",
    "will ",
    "would ",
    "shall ",
    "should ",
    "may ",
    "might ",
    "must ",
)

KNOWN_CHARACTER_GENDERS: dict[str, str] = {
    "dokyung": "male",
    "taejin": "male",
    "jinsang": "male",
    "hun": "male",
    "kyungsu": "male",
    "chairman": "male",
    "soontack": "male",
    "sangseok": "male",
    "gitae": "male",
    "yijoon": "male",
    "sungjin": "male",
    "chansu": "male",
    "jungtae": "male",
    "haeyoung1": "female",
    "haeyoung2": "female",
    "sukyung": "female",
    "jiya": "female",
    "heeran": "female",
    "deogi": "female",
    "anna": "female",
    "jeongsuk": "female",
    "youngji": "female",
    "bora": "female",
    "seohee": "female",
}

MALE_RELATION_SUBTYPES = {
    "boyfriend",
    "ex-boyfriend",
    "husband",
    "ex-husband",
    "son",
    "brother",
    "stepbrother",
    "father",
    "uncle",
    "nephew",
    "grandfather",
    "groom",
}

FEMALE_RELATION_SUBTYPES = {
    "girlfriend",
    "ex-girlfriend",
    "wife",
    "ex-wife",
    "daughter",
    "sister",
    "stepsister",
    "mother",
    "aunt",
    "niece",
    "grandmother",
    "bride",
}


def is_valid_kg_entity_name(name: str) -> bool:
    cleaned = " ".join(str(name).strip().split())
    if not cleaned or len(cleaned) > 40:
        return False
    lowered = cleaned.lower()
    invalid_prefixes = (
        "the ",
        "a ",
        "an ",
        "this ",
        "that ",
        "these ",
        "those ",
        "unnamed ",
        "unknown ",
        "someone",
        "somebody",
        "person ",
        "woman ",
        "man ",
    )
    if lowered.startswith(invalid_prefixes):
        return False
    if lowered in {
        "the woman",
        "the man",
        "the person",
        "woman",
        "man",
        "person",
        "he",
        "she",
        "they",
        "him",
        "her",
        "someone",
        "anyone",
        "null",
        "none",
    }:
        return False
    words = cleaned.split()
    if len(words) > 3:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9_.-]*(?: [A-Z][A-Za-z0-9_.-]*)*", cleaned))


def sanitize_kg_episodes(raw_episodes: Sequence[dict[str, Any]]) -> bool:
    changed = False
    for episode in raw_episodes:
        if not isinstance(episode, dict):
            continue
        knowledge = episode.get("episode_knowledge")
        if not isinstance(knowledge, dict):
            continue
        invalid_heads = [
            str(head) for head in list(knowledge) if not is_valid_kg_entity_name(str(head))
        ]
        for head in invalid_heads:
            knowledge.pop(head, None)
            changed = True
        for head, info in list(knowledge.items()):
            if not isinstance(info, dict):
                continue
            relationships = info.get("relationships")
            if not isinstance(relationships, dict):
                continue
            invalid_tails = [
                str(tail)
                for tail in list(relationships)
                if not is_valid_kg_entity_name(str(tail))
            ]
            for tail in invalid_tails:
                relationships.pop(tail, None)
                changed = True
    return changed


def format_gender_label(gender: str | None) -> str:
    if gender == "male":
        return "male (man, he/him)"
    if gender == "female":
        return "female (woman, she/her)"
    return "unknown"

