"""Core dataclasses, JSON/file helpers, and text/relation utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from q2q.config import TOKEN_RE


@dataclass
class QASample:
    question: str
    answer: str | None
    scene_ids: list[str]
    source: str
    shot_range: tuple[int, int] | None = None



@dataclass
class SceneRecord:
    scene_id: str
    episode_key: str
    knowledge: dict[str, Any] = field(default_factory=dict)



@dataclass
class Decision:
    done: bool
    sub_question: str | None = None
    final_answer: str | None = None
    evidence: list[str] = field(default_factory=list)



class GenerationCancelled(RuntimeError):
    """Raised when a web session requests cooperative generation cancellation."""



def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required data file was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc



def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write(chr(10))
    temporary_path.replace(path)



def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]



def meaningful_tokens(text: str) -> set[str]:
    stopwords = {
        "a", "an", "the", "and", "or", "but", "as", "at", "by", "for",
        "from", "in", "into", "of", "on", "to", "with", "who", "whom",
        "whose", "what", "where", "when", "why", "how", "which", "is",
        "are", "was", "were", "do", "does", "did", "has", "have", "had",
        "can", "could", "will", "would", "should", "may", "might", "must",
        "he", "she", "it", "they", "them", "his", "her", "their", "this",
        "that", "these", "those", "there", "then", "be", "been", "being",
    }
    return {token for token in tokenize(text) if len(token) > 1 and token not in stopwords}



def normalize_question(question: str) -> str:
    return " ".join(tokenize(question))



def normalize_binary_answer(value: str) -> str | None:
    normalized = value.strip().lower().rstrip(".!?")
    yes_values = {"yes", "y", "true", "1"}
    no_values = {"no", "n", "false", "0"}
    if normalized in yes_values:
        return "Yes"
    if normalized in no_values:
        return "No"
    return None



def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result



def episode_key_from_id(item_id: str) -> str:
    return item_id.split("_", 1)[0]



def format_gender_label(gender: str | None) -> str:
    if gender == "male":
        return "male (man, he/him)"
    if gender == "female":
        return "female (woman, she/her)"
    return "unknown"



def relation_pairs(relation_info: dict[str, Any]) -> list[tuple[str, str]]:
    relation_type = relation_info.get("type")
    subtype = relation_info.get("subtype")
    relation_types = (
        [relation_type] if isinstance(relation_type, str)
        else relation_type if isinstance(relation_type, list) else []
    )
    subtypes = (
        [subtype] if isinstance(subtype, str)
        else subtype if isinstance(subtype, list) else []
    )
    return [
        (str(relation_name), str(subtype_name))
        for relation_name, subtype_name in zip(relation_types, subtypes)
        if relation_name is not None and subtype_name is not None
    ]

