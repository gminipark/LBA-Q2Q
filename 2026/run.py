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


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def normalize_question(question: str) -> str:
    return " ".join(tokenize(question))


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


class DramaQAKnowledgeBase:
    """Loads DramaQA and provides exact-QA and lexical scene retrieval."""

    def __init__(
        self,
        kg_path: Path,
        qa_dir: Path | None = None,
        expanded_kg_path: Path | None = None,
        expansion_log_path: Path | None = None,
    ) -> None:
        self.kg_path = kg_path
        self.qa_dir = qa_dir
        self.expanded_kg_path = expanded_kg_path
        self.expansion_log_path = expansion_log_path
        self.episodes: dict[str, dict[str, Any]] = {}
        self.base_episodes: dict[str, dict[str, Any]] = {}
        self.scenes: dict[str, SceneRecord] = {}
        self.qa_samples: list[QASample] = []
        self.question_index: dict[str, list[str]] = defaultdict(list)
        self.entity_genders: dict[str, str] = {}
        self.added_relations_by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.verified_dialogue_history: list[dict[str, Any]] = []
        self._load_kg()
        self._load_external_qa()
        self._build_retrieval_index()

    def _add_qa(self, sample: QASample) -> None:
        self.qa_samples.append(sample)
        normalized = normalize_question(sample.question)
        if normalized:
            self.question_index[normalized].extend(sample.scene_ids)

    @staticmethod
    def _read_embedded_qa(
        raw: dict[str, Any], default_scene_ids: Sequence[str]
    ) -> tuple[str, str | None, list[str]] | None:
        question = raw.get("Question") or raw.get("question") or raw.get("que")
        if not isinstance(question, str) or not question.strip():
            return None
        answer = raw.get("Answer") or raw.get("answer")
        answer = answer if isinstance(answer, str) else None
        relevant = (
            raw.get("Relevant scene")
            or raw.get("relevant_scene")
            or raw.get("relevant_scenes")
            or default_scene_ids
        )
        if isinstance(relevant, str):
            scene_ids = [relevant]
        elif isinstance(relevant, list):
            scene_ids = [str(value) for value in relevant]
        else:
            scene_ids = list(default_scene_ids)
        return question.strip(), answer, unique_in_order(scene_ids)

    @staticmethod
    def _collect_episode_edge_set(episode: dict[str, Any]) -> set[tuple[str, str, str, str]]:
        edges: set[tuple[str, str, str, str]] = set()
        knowledge = episode.get("episode_knowledge", {})
        if not isinstance(knowledge, dict):
            return edges
        for head, info in knowledge.items():
            if not isinstance(info, dict):
                continue
            relationships = info.get("relationships", {})
            if not isinstance(relationships, dict):
                continue
            for tail, rel_info in relationships.items():
                if not isinstance(rel_info, dict):
                    continue
                r_type = rel_info.get("type")
                r_sub = rel_info.get("subtype")
                types = [r_type] if isinstance(r_type, str) else (r_type if isinstance(r_type, list) else [])
                subs = [r_sub] if isinstance(r_sub, str) else (r_sub if isinstance(r_sub, list) else [])
                for t_val, s_val in zip(types, subs):
                    if t_val and s_val:
                        edges.add((str(head), str(t_val), str(s_val), str(tail)))
        return edges

    def _infer_entity_genders(self) -> None:
        inferred: dict[str, str] = {}
        for episode in self.episodes.values():
            knowledge = episode.get("episode_knowledge", {})
            if not isinstance(knowledge, dict):
                continue
            for head, info in knowledge.items():
                head_str = str(head)
                if head_str.lower() in KNOWN_CHARACTER_GENDERS:
                    inferred[head_str] = KNOWN_CHARACTER_GENDERS[head_str.lower()]
                if not isinstance(info, dict):
                    continue
                relationships = info.get("relationships", {})
                if isinstance(relationships, dict):
                    for tail, rel_info in relationships.items():
                        tail_str = str(tail)
                        if tail_str.lower() in KNOWN_CHARACTER_GENDERS:
                            inferred[tail_str] = KNOWN_CHARACTER_GENDERS[tail_str.lower()]
                        if not isinstance(rel_info, dict):
                            continue
                        r_sub = rel_info.get("subtype")
                        subs = (
                            [r_sub]
                            if isinstance(r_sub, str)
                            else (r_sub if isinstance(r_sub, list) else [])
                        )
                        for sub in subs:
                            sub_lower = str(sub).strip().lower()
                            # In DramaQA KG, head -> tail: subtype means tail is head's subtype.
                            if sub_lower in MALE_RELATION_SUBTYPES and tail_str not in inferred:
                                inferred[tail_str] = "male"
                            elif sub_lower in FEMALE_RELATION_SUBTYPES and tail_str not in inferred:
                                inferred[tail_str] = "female"
        for scene in self.scenes.values():
            for entity in scene.knowledge:
                entity_str = str(entity)
                if entity_str.lower() in KNOWN_CHARACTER_GENDERS and entity_str not in inferred:
                    inferred[entity_str] = KNOWN_CHARACTER_GENDERS[entity_str.lower()]
        self.entity_genders = inferred

    def _load_kg(self) -> None:
        base_episodes = load_json(self.kg_path)
        if not isinstance(base_episodes, list):
            raise ValueError(f"KG root must be a list: {self.kg_path}")
        for ep in base_episodes:
            if isinstance(ep, dict):
                ep_key = episode_key_from_id(str(ep.get("episode", "")))
                if ep_key:
                    self.base_episodes[ep_key] = ep

        active_path = (
            self.expanded_kg_path
            if self.expanded_kg_path is not None and self.expanded_kg_path.exists()
            else self.kg_path
        )
        raw_episodes = load_json(active_path)
        if not isinstance(raw_episodes, list):
            raise ValueError(f"KG root must be a list: {active_path}")

        if sanitize_kg_episodes(raw_episodes) and active_path == self.expanded_kg_path:
            try:
                write_json_atomic(active_path, raw_episodes)
            except OSError:
                pass

        for episode in raw_episodes:
            if not isinstance(episode, dict):
                continue
            episode_id = str(episode.get("episode", ""))
            episode_key = episode_key_from_id(episode_id)
            if not episode_key:
                continue
            self.episodes[episode_key] = episode

            base_ep = self.base_episodes.get(episode_key)
            if base_ep is not None:
                base_edges = self._collect_episode_edge_set(base_ep)
                current_edges = self._collect_episode_edge_set(episode)
                for head, r_type, r_sub, tail in sorted(current_edges - base_edges):
                    self.added_relations_by_episode[episode_key].append(
                        {
                            "head": head,
                            "type": r_type,
                            "subtype": r_sub,
                            "tail": tail,
                            "fact": f"{tail} is {head}'s {r_sub} ({r_type})",
                        }
                    )

            history = episode.get("expanded_dialogue_history", [])
            if isinstance(history, list):
                for entry in history:
                    if isinstance(entry, dict):
                        self.verified_dialogue_history.append(entry)

            for scene_wrapper in episode.get("scene_level", []):
                if not isinstance(scene_wrapper, dict):
                    continue
                for scene_id, payload in scene_wrapper.items():
                    if not isinstance(payload, dict):
                        continue
                    scene = SceneRecord(
                        scene_id=str(scene_id),
                        episode_key=episode_key,
                        knowledge=payload.get("scene_knowledge", {})
                        if isinstance(payload.get("scene_knowledge", {}), dict)
                        else {},
                    )
                    self.scenes[scene.scene_id] = scene
                    for qa in payload.get("scene_qas", []):
                        if not isinstance(qa, dict):
                            continue
                        parsed = self._read_embedded_qa(qa, [scene.scene_id])
                        if parsed:
                            question, answer, scene_ids = parsed
                            self._add_qa(
                                QASample(
                                    question=question,
                                    answer=answer,
                                    scene_ids=scene_ids,
                                    source="KG/scene_qas",
                                )
                            )

            for qa_key in ("episode_qas", "episode_addqas"):
                for qa in episode.get(qa_key, []):
                    if not isinstance(qa, dict):
                        continue
                    parsed = self._read_embedded_qa(qa, [])
                    if parsed:
                        question, answer, scene_ids = parsed
                        self._add_qa(
                            QASample(
                                question=question,
                                answer=answer,
                                scene_ids=scene_ids,
                                source=f"KG/{qa_key}",
                            )
                        )

        if (
            not self.verified_dialogue_history
            and self.expansion_log_path is not None
            and self.expansion_log_path.exists()
        ):
            try:
                raw_log = load_json(self.expansion_log_path)
                if isinstance(raw_log, list):
                    turns: list[dict[str, Any]] = []
                    for item in raw_log:
                        if not isinstance(item, dict):
                            continue
                        q_text = str(item.get("question", "")).strip()
                        a_text = str(item.get("answer", "")).strip()
                        quad = item.get("quadruple")
                        if q_text and a_text:
                            turns.append(
                                {
                                    "sub_question": q_text,
                                    "answer": a_text,
                                    "quadruple": quad if isinstance(quad, dict) else None,
                                }
                            )
                    if turns:
                        self.verified_dialogue_history.append(
                            {
                                "original_question": "",
                                "final_answer": "",
                                "episode": "",
                                "turns": turns,
                            }
                        )
            except (OSError, ValueError):
                pass

        self._infer_entity_genders()

    def _load_external_qa(self) -> None:
        if self.qa_dir is None or not self.qa_dir.exists():
            return
        for path in sorted(self.qa_dir.glob("*.json")):
            raw_items = load_json(path)
            if not isinstance(raw_items, list):
                continue
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                question = item.get("que") or item.get("Question")
                scene_id = item.get("vid")
                if not isinstance(question, str) or not isinstance(scene_id, str):
                    continue

                answer: str | None = None
                answers = item.get("answers")
                correct_idx = item.get("correct_idx")
                if (
                    isinstance(answers, list)
                    and isinstance(correct_idx, int)
                    and 0 <= correct_idx < len(answers)
                    and isinstance(answers[correct_idx], str)
                ):
                    answer = answers[correct_idx]

                shot_range: tuple[int, int] | None = None
                shots = item.get("shot_contained")
                if (
                    isinstance(shots, list)
                    and len(shots) == 2
                    and all(isinstance(value, int) for value in shots)
                ):
                    shot_range = (min(shots), max(shots))

                self._add_qa(
                    QASample(
                        question=question.strip(),
                        answer=answer,
                        scene_ids=[scene_id],
                        source=f"QA/{path.name}",
                        shot_range=shot_range,
                    )
                )

    def _scene_retrieval_text(self, scene: SceneRecord) -> str:
        episode = self.episodes.get(scene.episode_key, {})
        pieces = [scene.scene_id]
        pieces.append(json.dumps(scene.knowledge, ensure_ascii=False))

        # Relationship names/types and occupations are important for questions such
        # as "What is X's relationship to Y?". They are included only in the
        # retrieval index; the final context is filtered separately.
        episode_knowledge = episode.get("episode_knowledge", {})
        if isinstance(episode_knowledge, dict):
            for entity, facts in episode_knowledge.items():
                pieces.append(str(entity))
                if isinstance(facts, dict):
                    pieces.append(str(facts.get("occupation", "")))
                    pieces.append(json.dumps(facts.get("relationships", {}), ensure_ascii=False))

        # Questions are useful retrieval anchors, but their reference answers are
        # deliberately not indexed or passed to the model.
        for sample in self.qa_samples:
            if scene.scene_id in sample.scene_ids:
                pieces.append(sample.question)
        return " ".join(pieces)

    def _build_retrieval_index(self) -> None:
        self._scene_ids = list(self.scenes)
        self._doc_tokens: list[list[str]] = []
        document_frequency: Counter[str] = Counter()
        for scene_id in self._scene_ids:
            tokens = tokenize(self._scene_retrieval_text(self.scenes[scene_id]))
            self._doc_tokens.append(tokens)
            document_frequency.update(set(tokens))

        document_count = max(len(self._doc_tokens), 1)
        self._idf = {
            token: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self._average_doc_length = (
            sum(len(tokens) for tokens in self._doc_tokens) / document_count
        ) or 1.0

    def find_exact_qa(self, question: str) -> QASample | None:
        normalized = normalize_question(question)
        for sample in self.qa_samples:
            if normalize_question(sample.question) == normalized:
                return sample
        return None

    def _record_for_id(self, scene_id: str) -> SceneRecord:
        if scene_id in self.scenes:
            return self.scenes[scene_id]
        # Train/validation QA covers episodes 1-15, while the provided KG covers
        # episodes 16-18. Preserve the ID so matching images can still be used.
        return SceneRecord(
            scene_id=scene_id,
            episode_key=episode_key_from_id(scene_id),
        )

    def retrieve(
        self,
        question: str,
        top_k: int,
        explicit_scene_ids: Sequence[str] | None = None,
    ) -> tuple[list[SceneRecord], str]:
        if explicit_scene_ids:
            ids = unique_in_order(str(value) for value in explicit_scene_ids)
            return [self._record_for_id(value) for value in ids[:top_k]], "--vid"

        exact_ids = unique_in_order(self.question_index.get(normalize_question(question), []))
        if exact_ids:
            return [self._record_for_id(value) for value in exact_ids[:top_k]], "exact QA mapping"

        query_tokens = tokenize(question)
        if not query_tokens or not self._scene_ids:
            return [], "no match"

        query_frequency = Counter(query_tokens)
        question_lower = question.lower()
        known_entities: set[str] = set()
        for episode in self.episodes.values():
            episode_knowledge = episode.get("episode_knowledge", {})
            if isinstance(episode_knowledge, dict):
                known_entities.update(str(entity) for entity in episode_knowledge)
        mentioned_entities = [
            entity for entity in known_entities if entity.lower() in question_lower
        ]
        k1 = 1.5
        b = 0.75
        scored: list[tuple[float, int]] = []
        for index, tokens in enumerate(self._doc_tokens):
            term_frequency = Counter(tokens)
            doc_length = max(len(tokens), 1)
            score = 0.0
            for term, query_count in query_frequency.items():
                frequency = term_frequency.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + k1 * (
                    1.0 - b + b * doc_length / self._average_doc_length
                )
                score += self._idf.get(term, 0.0) * (
                    frequency * (k1 + 1.0) / denominator
                ) * (1.0 + math.log(query_count))
            scene = self.scenes[self._scene_ids[index]]
            scene_entity_text = " ".join(str(key) for key in scene.knowledge).lower()
            matched_entities = sum(
                entity.lower() in scene_entity_text for entity in mentioned_entities
            )
            score += 8.0 * matched_entities
            if len(mentioned_entities) >= 2 and matched_entities == len(mentioned_entities):
                score += 12.0
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], self._scene_ids[item[1]]))
        records = [self.scenes[self._scene_ids[index]] for _, index in scored[:top_k]]
        return records, "BM25 KG retrieval"

    def _filtered_scene_knowledge(
        self,
        record: SceneRecord,
        question: str,
        selected_shot_ids: set[str],
    ) -> dict[str, Any]:
        if not isinstance(record.knowledge, dict):
            return {}
        query_terms = meaningful_tokens(question)
        ranked_entities: list[tuple[int, str, dict[str, Any]]] = []
        for entity, facts in record.knowledge.items():
            if not isinstance(facts, dict):
                continue
            entity_str = str(entity)
            fact_text = json.dumps(facts, ensure_ascii=False)
            # Base score of 1 ensures every character present in the retrieved scene
            # is visible to the model with their gender and scene actions/emotions.
            score = 1 + len(query_terms & meaningful_tokens(fact_text))
            for category in ("actions", "emotions"):
                values = facts.get(category, {})
                if isinstance(values, dict) and selected_shot_ids.intersection(values):
                    score += 8
                    break
            if entity_str.lower() in question.lower():
                score += 10
            ranked_entities.append((score, entity_str, facts))
        ranked_entities.sort(key=lambda item: (-item[0], item[1]))

        result: dict[str, Any] = {}
        for entity_score, entity, facts in ranked_entities[:6]:
            if entity_score <= 0:
                continue
            compact: dict[str, Any] = {
                "gender": format_gender_label(
                    self.entity_genders.get(entity)
                    or KNOWN_CHARACTER_GENDERS.get(entity.lower())
                )
            }
            for category in ("actions", "emotions"):
                values = facts.get(category, {})
                if not isinstance(values, dict) or not values:
                    continue
                ranked_values: list[tuple[int, str, Any]] = []
                for fact_id, value in values.items():
                    overlap = len(query_terms & meaningful_tokens(str(value)))
                    in_shot = 100 if fact_id in selected_shot_ids else 0
                    priority = in_shot + overlap * 5
                    ranked_values.append((priority, str(fact_id), value))
                ranked_values.sort(key=lambda item: (-item[0], item[1]))
                if ranked_values:
                    compact[category] = {
                        fact_id: value
                        for _, fact_id, value in ranked_values[:6]
                    }
            result[entity] = compact
        return result

    def _filtered_episode_knowledge(
        self, episode: dict[str, Any], scene_ids: set[str], question: str
    ) -> dict[str, Any]:
        knowledge = episode.get("episode_knowledge", {})
        if not isinstance(knowledge, dict):
            return {}

        question_lower = question.lower()
        selected_entities: set[str] = {
            str(entity)
            for entity in knowledge
            if is_valid_kg_entity_name(str(entity))
            and str(entity).lower() in question_lower
        }
        for scene_id in scene_ids:
            scene = self.scenes.get(scene_id)
            if scene:
                for entity in scene.knowledge:
                    if is_valid_kg_entity_name(str(entity)):
                        selected_entities.add(str(entity))

        # Expand seed entities via 1-hop graph relationships (outgoing and incoming).
        seed_entities = set(selected_entities)
        for entity in seed_entities:
            facts = knowledge.get(entity, {})
            if isinstance(facts, dict) and isinstance(facts.get("relationships"), dict):
                for target in facts["relationships"]:
                    target_str = str(target)
                    if is_valid_kg_entity_name(target_str):
                        selected_entities.add(target_str)
        for entity, facts in knowledge.items():
            entity_str = str(entity)
            if not is_valid_kg_entity_name(entity_str) or not isinstance(facts, dict):
                continue
            relationships = facts.get("relationships")
            if isinstance(relationships, dict) and any(
                str(target) in seed_entities for target in relationships
            ):
                selected_entities.add(entity_str)

        result: dict[str, Any] = {}
        for entity in sorted(selected_entities):
            facts = knowledge.get(entity)
            if not isinstance(facts, dict):
                continue
            gender = (
                self.entity_genders.get(entity)
                or KNOWN_CHARACTER_GENDERS.get(entity.lower())
            )
            compact: dict[str, Any] = {
                "gender": format_gender_label(gender),
            }
            if "occupation" in facts:
                compact["occupation"] = facts["occupation"]
            if "personality" in facts:
                compact["personality"] = facts["personality"]
            relationships = facts.get("relationships")
            if isinstance(relationships, dict):
                selected_relationships: dict[str, Any] = {}
                relationship_facts: list[str] = []
                for target, value in relationships.items():
                    target_str = str(target)
                    if not is_valid_kg_entity_name(target_str):
                        continue
                    if target_str not in selected_entities and len(selected_relationships) >= 6:
                        continue
                    selected_relationships[target_str] = value
                    if isinstance(value, dict):
                        for r_type, r_sub in relation_pairs(value):
                            relationship_facts.append(
                                f"{target_str} is {entity}'s {r_sub} ({r_type})"
                            )
                if selected_relationships:
                    compact["relationships_among_selected_entities"] = (
                        selected_relationships
                    )
                if relationship_facts:
                    compact["relationship_facts"] = relationship_facts
            result[entity] = compact
        return result

    def get_prior_verified_knowledge(
        self, question: str, records: Sequence[SceneRecord]
    ) -> dict[str, Any]:
        normalized_q = normalize_question(question)
        q_terms = meaningful_tokens(question)
        episode_keys = unique_in_order(record.episode_key for record in records)
        relevant_entities: set[str] = set()
        for record in records:
            for entity in record.knowledge:
                if is_valid_kg_entity_name(str(entity)):
                    relevant_entities.add(str(entity))
            episode = self.episodes.get(record.episode_key, {})
            ep_knowledge = episode.get("episode_knowledge", {})
            if isinstance(ep_knowledge, dict):
                for entity in ep_knowledge:
                    entity_str = str(entity)
                    if (
                        is_valid_kg_entity_name(entity_str)
                        and entity_str.lower() in question.lower()
                    ):
                        relevant_entities.add(entity_str)

        added_relations: list[dict[str, str]] = []
        for ep_key in episode_keys:
            for rel in self.added_relations_by_episode.get(ep_key, []):
                if (
                    rel["head"] in relevant_entities
                    or rel["tail"] in relevant_entities
                    or not relevant_entities
                ):
                    added_relations.append(rel)

        verified_subquestions: list[dict[str, Any]] = []
        seen_subq: set[str] = set()
        same_question_matched = False
        cached_final_answer: str | None = None

        for entry in self.verified_dialogue_history:
            entry_q = str(entry.get("original_question", "")).strip()
            entry_norm = normalize_question(entry_q)
            entry_terms = meaningful_tokens(entry_q)
            overlap_ratio = (
                len(q_terms & entry_terms) / max(len(q_terms), len(entry_terms), 1)
                if q_terms and entry_terms
                else 0.0
            )
            is_same_q = bool(
                entry_norm and (entry_norm == normalized_q or overlap_ratio >= 0.85)
            )
            if is_same_q:
                same_question_matched = True
                if entry.get("final_answer"):
                    cached_final_answer = str(entry["final_answer"]).strip()

            turns = entry.get("turns", [])
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                sub_q = str(turn.get("sub_question", "")).strip()
                ans = str(turn.get("answer", "")).strip()
                if not sub_q or not ans:
                    continue
                norm_sub = normalize_question(sub_q)
                if norm_sub in seen_subq:
                    continue
                sub_terms = meaningful_tokens(sub_q)
                if is_same_q or (q_terms and len(q_terms & sub_terms) >= 3):
                    seen_subq.add(norm_sub)
                    verified_subquestions.append(
                        {
                            "sub_question": sub_q,
                            "answer": ans,
                            "quadruple": (
                                turn.get("quadruple")
                                if isinstance(turn.get("quadruple"), dict)
                                else None
                            ),
                            "from_same_question": is_same_q,
                        }
                    )

        # Fallback for legacy logs without original_question: only match when a
        # "Who ..." question asks about an entity in the question and the expanded
        # relation connects that entity to the answer entity (which is not in the question).
        has_named_history = any(
            str(entry.get("original_question", "")).strip()
            for entry in self.verified_dialogue_history
            if isinstance(entry, dict)
        )
        if added_relations and not same_question_matched and not has_named_history:
            matched_qa = self.find_exact_qa(question)
            ref_answer = (matched_qa.answer or "").lower() if matched_qa else ""
            q_lower = question.lower().strip()
            if q_lower.startswith("who ") and ref_answer:
                for rel in added_relations:
                    head_l = rel["head"].lower()
                    tail_l = rel["tail"].lower()
                    if (
                        head_l in q_lower
                        and tail_l not in q_lower
                        and tail_l in ref_answer
                    ) or (
                        tail_l in q_lower
                        and head_l not in q_lower
                        and head_l in ref_answer
                    ):
                        same_question_matched = True
                        break

        return {
            "same_question_already_in_kg": same_question_matched,
            "cached_final_answer": cached_final_answer,
            "added_relations": added_relations,
            "verified_subquestions": verified_subquestions,
        }

    def build_context(
        self,
        question: str,
        records: Sequence[SceneRecord],
        selected_shot_ids: set[str] | None = None,
    ) -> str:
        selected_shot_ids = selected_shot_ids or set()
        selected_ids_by_episode: dict[str, set[str]] = defaultdict(set)
        scene_payloads: list[dict[str, Any]] = []
        for record in records:
            selected_ids_by_episode[record.episode_key].add(record.scene_id)
            scene_payloads.append(
                {
                    "scene_id": record.scene_id,
                    "scene_knowledge": self._filtered_scene_knowledge(
                        record, question, selected_shot_ids
                    ),
                }
            )

        episode_payloads: list[dict[str, Any]] = []
        for episode_key, scene_ids in selected_ids_by_episode.items():
            episode = self.episodes.get(episode_key)
            if not episode:
                episode_payloads.append(
                    {
                        "episode": episode_key,
                        "note": "This episode is not present in the provided test KG.",
                    }
                )
                continue
            ep_entry: dict[str, Any] = {
                "episode": episode.get("episode", episode_key),
                "relevant_entity_knowledge": self._filtered_episode_knowledge(
                    episode, scene_ids, question
                ),
            }
            if self.added_relations_by_episode.get(episode_key):
                ep_entry["expanded_kg_relations"] = self.added_relations_by_episode[
                    episode_key
                ]
            episode_payloads.append(ep_entry)

        prior_knowledge = self.get_prior_verified_knowledge(question, records)
        payload: dict[str, Any] = {
            "kg_schema_note": (
                "Always check each character's 'gender', 'occupation', 'personality', "
                "'relationship_facts', and 'scene_knowledge' before generating a "
                "sub-question. In relationships, 'X is Y's <subtype>' states X's role "
                "to Y (e.g. 'Haeyoung1 is Dokyung's girlfriend' means Dokyung is male "
                "and Haeyoung1 is female). Never misgender a character or ask a "
                "sub-question whose answer is already verified in the KG."
            ),
            "selected_scenes": scene_payloads,
            "episode_context": episode_payloads,
        }
        if (
            prior_knowledge["verified_subquestions"]
            or prior_knowledge["added_relations"]
        ):
            payload["already_verified_in_kg_do_not_repeat"] = {
                "same_question_already_in_kg": prior_knowledge[
                    "same_question_already_in_kg"
                ],
                "expanded_kg_relations": prior_knowledge["added_relations"],
                "previously_verified_subquestions": prior_knowledge[
                    "verified_subquestions"
                ],
            }
        return json.dumps(payload, ensure_ascii=False, indent=2)


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


def is_plausible_english_question(text: str) -> bool:
    question = " ".join(text.strip().split())
    if not question.isascii() or not 6 <= len(question) <= 220:
        return False
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\s,.\x27?!():-]*", question) is None:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9\x27-]*", question)
    if not 3 <= len(words) <= 32:
        return False
    return all(len(word) <= 30 for word in words)



def is_yes_no_question(text: str) -> bool:
    question = " ".join(text.strip().split())
    if not is_plausible_english_question(question):
        return False
    if not question.endswith("?") or question.count("?") != 1:
        return False
    lowered = question.lower()
    if lowered.startswith(ENGLISH_WH_PREFIXES):
        return False
    if lowered.startswith(ENGLISH_BINARY_PREFIXES):
        return True
    return False


def rewrite_ambiguous_entity_clauses(
    question: str,
    entity_genders: dict[str, str] | None = None,
) -> str:
    merged_genders = {
        **KNOWN_CHARACTER_GENDERS,
        **{k.lower(): v for k, v in (entity_genders or {}).items()},
    }
    relative_Subject_verbs = (
        "meets|met|runs|ran|sees|saw|talks|talked|speaks|spoke|greets|greeted|"
        "bumps|bumped|visits|visited|calls|called|helps|helped|knows|knew|"
        "loves|loved|likes|liked|dates|dated|marries|married|gives|gave|"
        "hands|handed|approaches|approached|confronts|confronted|follows|followed|"
        "watches|watched|Hugs|hugged|kisses|kissed|pushes|pushed|pulls|pulled|"
        "notices|noticed|recognizes|recognized|encounters|encountered"
    )
    # Rewrite reduced relative clauses like "woman Dokyung meets" -> "woman whom Dokyung meets"
    # so a generic gender noun and a character name are never juxtaposed as "woman Dokyung".
    pattern = re.compile(
        rf"\b(woman|man|person|girl|boy|lady|guy|colleague|coworker|employee|stranger)\s+"
        rf"([A-Z][A-Za-z0-9_]*)\s+({relative_Subject_verbs})\b",
        flags=re.I,
    )
    question = pattern.sub(r"\1 whom \2 \3", question)

    # If the model literally wrote "the woman <MaleName>" or "the man <FemaleName>"
    # as an appositive, strip the wrong gender noun.
    def _fix_appositive(match: re.Match[str]) -> str:
        det = match.group(1) or ""
        noun = match.group(2)
        name = match.group(3)
        gender = merged_genders.get(name.lower())
        noun_lower = noun.lower()
        if gender == "male" and noun_lower in {"woman", "female", "girl", "lady"}:
            return name
        if gender == "female" and noun_lower in {"man", "male", "boy", "guy"}:
            return name
        return f"{det}{noun} {name}"

    question = re.sub(
        r"\b(the\s+|a\s+|an\s+)?(woman|man|female|male|girl|boy|lady|guy)\s+([A-Z][A-Za-z0-9_]*)\b(?!\s+(?:whom|who|that)\b)",
        _fix_appositive,
        question,
        flags=re.I,
    )
    return " ".join(question.split())


def normalize_model_sub_question(
    text: str,
    entity_genders: dict[str, str] | None = None,
) -> str | None:
    question = " ".join(text.strip().strip("\"").split())
    question = re.sub(
        r"^(?:sub[- ]?question|question)\s*:\s*", "", question, flags=re.I
    )
    if "?" in question:
        question = question.split("?", 1)[0].strip() + "?"
    elif question:
        question = question.rstrip(".! ") + "?"
    question = rewrite_ambiguous_entity_clauses(question, entity_genders)
    if is_yes_no_question(question):
        return question
    if not is_plausible_english_question(question):
        return None

    lowered = question.lower()
    non_binary_prefixes = (
        "tell ", "name ", "list ", "explain ", "describe ", "identify "
    )
    if not question or lowered.startswith(ENGLISH_WH_PREFIXES + non_binary_prefixes):
        return None

    statement = question[:-1].strip()
    statement_terms = set(tokenize(statement))
    binary_verbs = {prefix.strip() for prefix in ENGLISH_BINARY_PREFIXES}
    if not statement or not (statement_terms & binary_verbs):
        return None
    candidate = f"Is it true that {statement[:1].lower() + statement[1:]}?"
    candidate = rewrite_ambiguous_entity_clauses(candidate, entity_genders)
    return candidate if is_yes_no_question(candidate) else None


def safe_fallback_sub_question(asked_questions: Sequence[str]) -> str:
    candidates = (
        "Does the selected visual evidence show the relevant event?",
        "Does the retrieved KG describe the relevant event?",
        "Can a visible action distinguish the relevant entities?",
        "Does a visible interaction distinguish the relevant entities?",
        "Can an indirect KG relation distinguish the relevant entities?",
        "Does the selected evidence reveal a useful non-target attribute?",
    )
    asked = {normalize_question(item) for item in asked_questions}
    for candidate in candidates:
        if normalize_question(candidate) not in asked:
            return candidate
    return candidates[-1]



def json_objects_from_text(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def parse_decision(text: str) -> Decision:
    visible_text = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    objects = json_objects_from_text(visible_text)
    if objects:
        value = objects[-1]
        done = value.get("done")
        if isinstance(done, str):
            done = done.strip().lower() == "true"
        if isinstance(done, bool):
            if done:
                final_answer = value.get("final_answer") or value.get("answer")
                evidence = value.get("evidence", [])
                if isinstance(final_answer, str) and final_answer.strip():
                    return Decision(
                        done=True,
                        final_answer=final_answer.strip(),
                        evidence=[str(item) for item in evidence]
                        if isinstance(evidence, list)
                        else [],
                    )
            else:
                sub_question = value.get("sub_question") or value.get("question")
                if isinstance(sub_question, str) and sub_question.strip():
                    cleaned_sub = rewrite_ambiguous_entity_clauses(sub_question.strip())
                    return Decision(done=False, sub_question=cleaned_sub)

    # Gracefully accept a bare binary question if JSON formatting alone failed.
    candidates = [line.strip() for line in visible_text.splitlines() if line.strip().endswith("?")]
    for candidate in reversed(candidates):
        candidate = re.sub(r"^(sub[- ]?question\s*:\s*)", "", candidate, flags=re.I)
        candidate = rewrite_ambiguous_entity_clauses(candidate)
        if is_yes_no_question(candidate):
            return Decision(done=False, sub_question=candidate)
    raise ValueError("The model response did not contain a valid decision JSON object.")


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


def kg_entity_consistency_reason(
    sub_question: str,
    entity_genders: dict[str, str] | None = None,
) -> str | None:
    merged_genders = {
        **KNOWN_CHARACTER_GENDERS,
        **{k.lower(): v for k, v in (entity_genders or {}).items()},
    }
    lowered = sub_question.lower()
    tokens = tokenize(sub_question)
    mentioned_males = [
        name for name, gender in merged_genders.items()
        if gender == "male" and re.search(rf"\b{re.escape(name)}\b", lowered)
    ]
    mentioned_females = [
        name for name, gender in merged_genders.items()
        if gender == "female" and re.search(rf"\b{re.escape(name)}\b", lowered)
    ]

    for male_name in mentioned_males:
        # Reject phrases like "woman Dokyung", "female Dokyung", "girl Dokyung", "is Dokyung a woman/girlfriend"
        if re.search(
            rf"\b(?:woman|female|girl|lady)\s+{re.escape(male_name)}\b"
            rf"|\b{re.escape(male_name)}\s+is\s+(?:a\s+|an\s+|the\s+)?(?:woman|female|girl|lady|girlfriend|wife|mother|sister|daughter)\b"
            rf"|\bis\s+{re.escape(male_name)}\s+(?:a\s+|an\s+|the\s+)?(?:woman|female|girl|lady|girlfriend|wife|mother|sister|daughter)\b",
            lowered,
        ):
            return (
                f"According to the KG, '{male_name.capitalize()}' is male (man, he/him), "
                f"not a woman/female. Check the KG character gender and relationships "
                f"carefully and do not write 'woman {male_name.capitalize()}'."
            )
        # If this male character is the only person mentioned (no female character or generic woman/person noun),
        # reject female pronouns 'she'/'her'/'herself'.
        has_other_person_reference = bool(mentioned_females) or any(
            word in tokens
            for word in (
                "woman", "girl", "lady", "female", "person", "someone",
                "colleague", "coworker", "employee", "mother", "sister", "girlfriend",
            )
        )
        if not has_other_person_reference and any(
            pron in tokens for pron in ("she", "her", "herself")
        ):
            return (
                f"According to the KG, '{male_name.capitalize()}' is male (he/him). "
                "Do not refer to a male character with female pronouns (she/her)."
            )

    for female_name in mentioned_females:
        if re.search(
            rf"\b(?:man|male|boy|guy)\s+{re.escape(female_name)}\b"
            rf"|\b{re.escape(female_name)}\s+is\s+(?:a\s+|an\s+|the\s+)?(?:man|male|boy|guy|boyfriend|husband|father|brother|son)\b"
            rf"|\bis\s+{re.escape(female_name)}\s+(?:a\s+|an\s+|the\s+)?(?:man|male|boy|guy|boyfriend|husband|father|brother|son)\b",
            lowered,
        ):
            return (
                f"According to the KG, '{female_name.capitalize()}' is female (woman, she/her), "
                f"not a man/male. Check the KG character gender and relationships carefully."
            )
        has_other_person_reference = bool(mentioned_males) or any(
            word in tokens
            for word in (
                "man", "boy", "guy", "male", "person", "someone",
                "colleague", "coworker", "employee", "father", "brother", "boyfriend",
            )
        )
        if not has_other_person_reference and any(
            pron in tokens for pron in ("he", "him", "his", "himself")
        ):
            return (
                f"According to the KG, '{female_name.capitalize()}' is female (she/her). "
                "Do not refer to a female character with male pronouns (he/him/his)."
            )
    return None


def already_known_in_kg_reason(
    sub_question: str,
    prior_kg_questions: Sequence[str] = (),
    added_relations: Sequence[dict[str, str]] = (),
    can_finish_now: bool = False,
) -> str | None:
    if not prior_kg_questions and not added_relations:
        return None
    normalized_sub = normalize_question(sub_question)
    sub_terms = meaningful_tokens(sub_question)
    finish_hint = (
        "Since the required knowledge is already in the KG and the minimum sub-question "
        "requirement is met, return {\"done\": true, \"final_answer\": \"...\", \"evidence\": [...]} now."
        if can_finish_now
        else "Ask a different binary sub-question about an unverified detail not already in the KG."
    )

    for prior_q in prior_kg_questions:
        norm_prior = normalize_question(prior_q)
        if not norm_prior:
            continue
        if normalized_sub == norm_prior:
            return (
                f"The sub-question '{prior_q}' was already verified in a prior session and "
                f"its knowledge is already in the KG. Do not repeat it. {finish_hint}"
            )
        prior_terms = meaningful_tokens(prior_q)
        if sub_terms and prior_terms:
            overlap = len(sub_terms & prior_terms)
            min_len = max(min(len(sub_terms), len(prior_terms)), 1)
            if overlap >= 2 and (overlap / min_len) >= 0.72:
                return (
                    f"A very similar sub-question ('{prior_q}') was already verified "
                    f"and stored in the KG. Do not ask redundant questions about knowledge "
                    f"already in the graph. {finish_hint}"
                )

    for rel in added_relations:
        r_type = str(rel.get("type", "")).lower()
        r_sub = str(rel.get("subtype", "")).lower()
        head = str(rel.get("head", "")).lower()
        tail = str(rel.get("tail", "")).lower()
        rel_synonyms: set[str] = set(tokenize(f"{r_type} {r_sub}"))
        if r_sub in {"colleague", "coworker"} or r_type == "work":
            rel_synonyms.update({"colleague", "colleagues", "coworker", "coworkers", "company", "work", "office"})
        if r_sub in {"girlfriend", "boyfriend", "lover"} or r_type == "romantic":
            rel_synonyms.update({"girlfriend", "boyfriend", "romantic", "dating", "lover"})
        if r_sub in {"friend", "acquaintance"}:
            rel_synonyms.update({"friend", "friends", "acquaintance"})
        if (head in sub_terms or tail in sub_terms) and (sub_terms & rel_synonyms):
            return (
                f"The KG already contains the expanded relation '{rel.get('fact', '')}'. "
                f"Do not re-ask a sub-question to verify a relation already in the KG. {finish_hint}"
            )
    return None


def direct_answer_probe_reason(
    original_question: str,
    sub_question: str,
    reference_answer: str | None,
) -> str | None:
    original_terms = meaningful_tokens(original_question)
    original_lower = original_question.lstrip().lower()
    sub_terms = meaningful_tokens(sub_question)
    if reference_answer:
        answer_only_terms = meaningful_tokens(reference_answer) - original_terms
        if answer_only_terms & sub_terms:
            return (
                "The sub-question directly mentions information unique to the "
                "reference answer. Ask an indirect evidence question about a visible "
                "attribute, action, location, or non-target KG relation without "
                "naming or proposing the final answer."
            )
    elif original_lower.startswith(("who ", "whom ", "whose ")):
        original_surface = set(TOKEN_RE.findall(original_question))
        new_named_candidates = {
            token
            for token in TOKEN_RE.findall(sub_question)
            if (token[:1].isupper() or any(char.isdigit() for char in token))
            and token not in original_surface
            and token.lower() not in {"is", "are", "does", "did", "can", "kg"}
        }
        if new_named_candidates:
            return (
                "A Who-question sub-question must not introduce a new named answer "
                "candidate. Ask about unnamed visual or KG evidence instead."
            )
    elif original_lower.startswith("where "):
        spatial_prepositions = {"in", "at", "on", "near", "inside", "outside"}
        if spatial_prepositions & set(tokenize(sub_question)) and (sub_terms - original_terms):
            return (
                "A Where-question sub-question must not propose a specific final "
                "location. Ask only coarse or upstream scene evidence."
            )
    elif "relationship" in original_lower or "relation between" in original_lower:
        target_relations = {
            "romantic", "family", "friend", "friends", "friendship", "coworker",
            "coworkers", "colleague", "colleagues", "rival", "rivals", "sibling",
            "siblings", "parent", "child", "married", "dating", "estranged",
        }
        if target_relations & sub_terms:
            return (
                "A relationship-target question must not propose the final relation "
                "label. Ask about an indirect interaction or shared context."
            )
    elif original_terms:
        coverage = len(original_terms & sub_terms) / len(original_terms)
        if coverage >= 0.8 and (sub_terms - original_terms):
            return (
                "The sub-question is a binary reformulation of the original target. "
                "Ask an upstream evidence question instead of proposing its answer."
            )
    return None


def validate_decision(
    decision: Decision,
    asked_questions: Sequence[str],
    min_subquestions: int,
    force_final: bool,
    original_question: str = "",
    reference_answer: str | None = None,
    entity_genders: dict[str, str] | None = None,
    prior_kg_questions: Sequence[str] = (),
    added_relations: Sequence[dict[str, str]] = (),
    answers: Sequence[str] = (),
) -> str | None:
    if force_final:
        if not decision.done or not decision.final_answer:
            return "You must return done=true and a non-empty final_answer now."
        return None
    if decision.done:
        if len(asked_questions) < min_subquestions:
            return (
                f"You must ask at least {min_subquestions} binary sub-questions before "
                "returning the final answer."
            )
        if asked_questions and answers and "Yes" not in answers:
            return (
                "None of your sub-questions have been confirmed with 'Yes' yet (all answers "
                "so far were 'No'). Ask a follow-up binary sub-question to verify an "
                "alternative clue before returning the final answer."
            )
        if not decision.final_answer or not decision.final_answer.strip():
            return "When done=true, you must provide a non-empty final_answer."
        return None
    if not decision.sub_question or not is_yes_no_question(decision.sub_question):
        return (
            "Return exactly one yes/no sub-question. It must end in one '?' and must "
            "use ASCII English, and not begin with a wh-word such as what, where, who, why, or how."
        )
    gender_error = kg_entity_consistency_reason(
        decision.sub_question, entity_genders
    )
    if gender_error:
        return gender_error
    direct_probe_error = direct_answer_probe_reason(
        original_question, decision.sub_question, reference_answer
    )
    if direct_probe_error:
        return direct_probe_error
    normalized = normalize_question(decision.sub_question)
    if normalized in {normalize_question(item) for item in asked_questions}:
        return "Do not repeat an earlier sub-question; ask a new binary question."
    kg_repeat_error = already_known_in_kg_reason(
        decision.sub_question,
        prior_kg_questions=prior_kg_questions,
        added_relations=added_relations,
        can_finish_now=(
            len(asked_questions) >= min_subquestions
            and ("Yes" in answers or not answers)
        ),
    )
    if kg_repeat_error:
        return kg_repeat_error
    return None


def build_turn_followup_prompt(
    answer: str,
    remaining: int,
    asked_count: int,
    min_subquestions: int,
    answers_so_far: Sequence[str],
) -> str:
    remaining = max(remaining, 0)
    can_finish_now = (
        asked_count >= min_subquestions
        and ("Yes" in answers_so_far or not answers_so_far)
    )
    if answer == "No":
        if not can_finish_now:
            return (
                f"Answer to that sub-question: No. "
                f"Because the answer is No, that hypothesis was rejected and no positive "
                f"clue has been confirmed yet. You have {remaining} sub-question(s) remaining. "
                "Ask a follow-up non-repeated yes/no sub-question to test an alternative clue "
                "({\"done\": false, \"sub_question\": \"...\"})."
            )
        return (
            f"Answer to that sub-question: No. "
            f"You have {remaining} sub-question(s) remaining. "
            "Evaluate whether your previously confirmed clues plus this 'No' already uniquely "
            "determine the answer to the ORIGINAL QUESTION: if another clue still needs to be "
            "verified, ask a follow-up non-repeated yes/no sub-question "
            "({\"done\": false, \"sub_question\": \"...\"}); otherwise return "
            "{\"done\": true, \"final_answer\": \"...\", \"evidence\": [...]}."
        )
    if not can_finish_now:
        needed = max(min_subquestions - asked_count, 1)
        return (
            f"Answer to that sub-question: {answer}. "
            f"You must ask at least {needed} more sub-question(s) (up to {remaining} remaining). "
            "Ask the next non-repeated yes/no sub-question."
        )
    return (
        f"Answer to that sub-question: {answer}. "
        f"You may ask up to {remaining} more sub-question(s) ONLY IF more information is needed. "
        "Carefully check whether the RETRIEVED KG EVIDENCE, visual frames, and confirmed "
        "dialogue answer(s) now suffice to answer every part of the ORIGINAL QUESTION: "
        "- If the answer is now clear and determined, finish immediately by returning "
        "{\"done\": true, \"final_answer\": \"...\", \"evidence\": [...]}. "
        "- If another part of the ORIGINAL QUESTION or an ambiguity between candidates "
        "still requires verification, ask one more non-repeated yes/no sub-question "
        "({\"done\": false, \"sub_question\": \"...\"})."
    )


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def system_prompt(min_subquestions: int, max_subquestions: int) -> str:
    min_rule = (
        f"7. Ask at least {min_subquestions} and at most {max_subquestions} sub-questions. "
        "Decide flexibly after each answer whether you have enough information: as soon as "
        "the RETRIEVED KG EVIDENCE, video frames, and confirmed dialogue answer(s) "
        "uniquely determine the full answer to the ORIGINAL QUESTION (with at least one "
        "confirmed 'Yes' clue), return {\"done\": true, \"final_answer\": \"...\", "
        "\"evidence\": [...]} without asking unnecessary extra questions. If an answer is "
        "'No' or if part of the ORIGINAL QUESTION / candidate disambiguation still lacks "
        "information, ask an additional yes/no sub-question."
        if min_subquestions > 0
        else (
            f"7. Prior KG verification already satisfies the minimum sub-question count "
            f"(0 additional sub-questions required, at most {max_subquestions} allowed). "
            "If RETRIEVED KG EVIDENCE (including 'already_verified_in_kg_do_not_repeat' "
            "and 'relevant_entity_knowledge') already determines the answer to the "
            "ORIGINAL QUESTION, return {\"done\": true, \"final_answer\": \"...\", "
            "\"evidence\": [...]} immediately without asking redundant sub-questions."
        )
    )
    return f"""You are an evidence-grounded sub-questioning agent for DramaQA.

Your goal is to answer the ORIGINAL QUESTION by conducting a short interactive
dialogue. You MUST carefully inspect the supplied RETRIEVED KG EVIDENCE (character
genders, occupations, personalities, relationship_facts, scene_knowledge, and any
already_verified_in_kg_do_not_repeat entries) together with the video frames.

Rules:
1. Ask exactly ONE sub-question per turn (unless done=true is already warranted).
2. Every sub-question MUST be answerable only with yes or no. Use ASCII English
   only and keep it under 25 words.
3. Strictly respect KG character genders and relationships:
   - Check each character's 'gender' field in RETRIEVED KG EVIDENCE (e.g., Dokyung is
     male: man/he/him; Haeyoung1 and Haeyoung2 are female: woman/she/her).
   - In KG relationships, "X is Y's <subtype>" states what X is to Y (e.g.,
     "Haeyoung1 is Dokyung's girlfriend" means Haeyoung1 is female and Dokyung is male).
   - Never misgender a character (never write "woman Dokyung" or "man Haeyoung1").
   - When referring to an unnamed person whom a character meets, always include the
     relative pronoun "whom" or "who" (e.g., "the woman whom Dokyung meets", NEVER
     "the woman Dokyung meets").
4. Never ask an open wh-question. Convert hypotheses such as "where is X?" into
   binary alternatives such as "Is X in the office?" Convert relationship checks
   into questions such as "Are X and Y coworkers?"
5. Never ask whether a proposed final answer or answer candidate is correct. Do not
   turn the original Who/Where/What/Which/How target into a yes/no guess.
6. Ask only upstream evidence questions about visible attributes, actions, locations,
   or non-target KG relations. A yes answer must not itself reveal the final answer.
{min_rule}
8. NEVER ask a sub-question whose answer is already stated in RETRIEVED KG EVIDENCE
   or listed under 'already_verified_in_kg_do_not_repeat'. If the same question comes
   in again and the needed facts/relations are already in the KG, do not repeat the
   same sub-questions.
9. Keep internal reasoning brief so that the required JSON is always emitted.
10. Do not reveal chain-of-thought. Return only one JSON object outside any internal
   reasoning.

Allowed indirect examples:
- Original target: "Who did Dokyung meet?" -> "Does the woman whom Dokyung meets work at Haeyoung1's company?"
- Original target: "Where is X?" -> "Is X indoors?"
- Original target: "What is X and Y relationship?" -> "Do X and Y work at the same place?"
Forbidden direct guesses: "Is the person Haeyoung2?" or "Is X in the hospital?"

For a sub-question, return exactly:
{{"done": false, "sub_question": "Is/Are/Does/Did/Can ...?"}}

When enough information is available, return exactly:
{{"done": true, "final_answer": "...", "evidence": ["short evidence item"]}}
"""


def initial_messages(
    question: str,
    context: str,
    images: Sequence[tuple[str, Path]],
    min_subquestions: int,
    max_subquestions: int,
    prior_knowledge: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    for index, (scene_id, path) in enumerate(images, start=1):
        user_content.append(
            {
                "type": "text",
                "text": f"Representative frame {index} from scene {scene_id}:",
            }
        )
        user_content.append({"type": "image", "image": str(path)})
    if not images:
        user_content.append(
            {
                "type": "text",
                "text": "No matching image frame was available; rely on KG evidence.",
            }
        )

    same_in_kg = bool(
        prior_knowledge and prior_knowledge.get("same_question_already_in_kg")
    )
    verified_list = (
        prior_knowledge.get("verified_subquestions", [])
        if isinstance(prior_knowledge, dict)
        else []
    )
    added_rels = (
        prior_knowledge.get("added_relations", [])
        if isinstance(prior_knowledge, dict)
        else []
    )

    if min_subquestions <= 0 and (same_in_kg or verified_list or added_rels):
        closing_instruction = (
            "IMPORTANT: This question (or its required relational evidence) has ALREADY "
            "been verified and stored in the Knowledge Graph (see "
            "'already_verified_in_kg_do_not_repeat' and 'relevant_entity_knowledge' above). "
            "Do NOT ask the same sub-questions again. Because the needed knowledge is "
            "already in the KG, return {\"done\": true, \"final_answer\": \"...\", "
            "\"evidence\": [...]} immediately grounded in the KG evidence."
        )
    elif verified_list or added_rels:
        closing_instruction = (
            "Begin the dialogue. Carefully check character genders and existing relations "
            "in RETRIEVED KG EVIDENCE, and do NOT repeat any question or relation listed "
            "in 'already_verified_in_kg_do_not_repeat'. Return only the JSON object for "
            "the first NEW yes/no sub-question."
        )
    else:
        closing_instruction = (
            "Begin the dialogue. Carefully check character genders, occupations, "
            "personalities, relationships, and scene_knowledge in RETRIEVED KG EVIDENCE "
            "before formulating your sub-question. Return only the JSON object for the "
            "first yes/no sub-question. Do not answer the original question yet."
        )

    user_content.append(
        {
            "type": "text",
            "text": (
                f"ORIGINAL QUESTION:\n{question}\n\n"
                f"RETRIEVED KG EVIDENCE:\n{context}\n\n"
                f"{closing_instruction}"
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": text_content(system_prompt(min_subquestions, max_subquestions)),
        },
        {"role": "user", "content": user_content},
    ]


class QwenSubQuestionAgent:
    def __init__(
        self,
        model_name: str,
        dtype: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        attn_implementation: str | None,
        cache_dir: Path | None,
        local_files_only: bool,
        original_question: str,
        reference_answer: str | None,
        max_gpu_memory_gib: float | None = None,
        max_cpu_memory_gib: float | None = None,
        entity_genders: dict[str, str] | None = None,
        prior_kg_questions: Sequence[str] = (),
        added_relations: Sequence[dict[str, str]] = (),
        same_question_in_kg: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen multimodal inference requires torch, transformers, "
                "accelerate, torchvision, and pillow. "
                f"(python={sys.executable}, error={exc})"
            ) from exc

        model_name_lower = model_name.lower()
        is_qwen35 = any(
            marker in model_name_lower
            for marker in (
                "qwen3.5",
                "qwen3_5",
                "qwen3.6",
                "qwen3_6",
                "qwen3.8",
                "qwen3_8",
            )
        )
        try:
            if is_qwen35:
                from transformers import Qwen3_5ForConditionalGeneration

                model_class = Qwen3_5ForConditionalGeneration
            else:
                from transformers import Qwen3VLForConditionalGeneration

                model_class = Qwen3VLForConditionalGeneration
        except ImportError as exc:
            requirement = (
                "the latest Transformers main branch"
                if is_qwen35
                else "transformers>=4.57.0"
            )
            raise RuntimeError(
                f"{model_name} requires {requirement}. Activate the LBA Conda "
                "environment before running this script. "
                f"(python={sys.executable}, error={exc})"
            ) from exc

        dtype_map = {
            "auto": "auto",
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        model_kwargs: dict[str, Any] = {
            "dtype": dtype_map[dtype],
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            model_kwargs["cache_dir"] = str(cache_dir)
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if max_gpu_memory_gib is not None or max_cpu_memory_gib is not None:
            max_memory: dict[int | str, str] = {}
            if max_gpu_memory_gib is not None:
                if max_gpu_memory_gib <= 0:
                    raise ValueError("GPU memory limit must be greater than zero.")
                gpu_count = torch.cuda.device_count()
                if gpu_count == 0:
                    raise RuntimeError(
                        "A GPU memory limit was requested, but no visible CUDA GPU "
                        "was found. Check CUDA_VISIBLE_DEVICES."
                    )
                gpu_limit = f"{max_gpu_memory_gib:g}GiB"
                max_memory.update(
                    {device_index: gpu_limit for device_index in range(gpu_count)}
                )
            if max_cpu_memory_gib is not None:
                if max_cpu_memory_gib <= 0:
                    raise ValueError("CPU memory limit must be greater than zero.")
                max_memory["cpu"] = f"{max_cpu_memory_gib:g}GiB"
            model_kwargs["max_memory"] = max_memory

        print(f"[model] Loading {model_name} ...", flush=True)
        self.model = model_class.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        processor_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            processor_kwargs["cache_dir"] = str(cache_dir)
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.torch = torch
        self.is_qwen35 = is_qwen35
        enable_thinking = os.environ.get("LBA_ENABLE_THINKING", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self.requires_complete_thinking = (
            enable_thinking and (is_qwen35 or "thinking" in model_name_lower)
        )
        self.chat_template_kwargs = (
            {"enable_thinking": enable_thinking} if is_qwen35 else {}
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.original_question = original_question
        self.reference_answer = reference_answer
        self.entity_genders: dict[str, str] = dict(entity_genders or {})
        self.prior_kg_questions: list[str] = list(prior_kg_questions)
        self.added_relations: list[dict[str, str]] = list(added_relations)
        self.same_question_in_kg: bool = bool(same_question_in_kg)
        self.stop_event: Event | None = None

    def _raise_if_cancelled(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise GenerationCancelled("Generation was cancelled by the user.")

    def _run_model_generate(
        self, inputs: Any, **generation_kwargs: Any
    ) -> Any:
        self._raise_if_cancelled()
        if self.stop_event is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList

            stop_event = self.stop_event

            class EventStoppingCriteria(StoppingCriteria):
                def __call__(
                    self, input_ids: Any, scores: Any, **kwargs: Any
                ) -> bool:
                    return stop_event.is_set()

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [EventStoppingCriteria()]
            )
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        self._raise_if_cancelled()
        return generated

    def generate(self, messages: Sequence[dict[str, Any]]) -> str:
        inputs = self.processor.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self.chat_template_kwargs,
        )
        inputs = inputs.to(self.model.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        generated = self._run_model_generate(inputs, **generation_kwargs)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated)
        ]
        raw_with_special = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if "</think>" in raw_with_special and "</think>" not in decoded:
            think_part, visible_part = raw_with_special.rsplit("</think>", 1)
            special_tokens = getattr(
                getattr(self.processor, "tokenizer", None),
                "all_special_tokens",
                [],
            )
            for tok in special_tokens:
                if tok != "</think>":
                    think_part = think_part.replace(tok, "")
                    visible_part = visible_part.replace(tok, "")
            return f"{think_part.strip()}\n</think>\n\n{visible_part.strip()}"
        return decoded

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        enable_thinking: bool = False,
    ) -> str:
        template_kwargs = (
            {"enable_thinking": enable_thinking} if self.is_qwen35 else {}
        )
        messages = [
            {"role": "user", "content": text_content(prompt)}
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        inputs = inputs.to(self.model.device)
        generated = self._run_model_generate(
            inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


    def complete_structured_output(
        self,
        messages: Sequence[dict[str, Any]],
        reasoning: str,
        force_final: bool,
        can_finish: bool = False,
    ) -> str:
        print(
            "[model] Completing structured JSON output ...",
            file=sys.stderr,
            flush=True,
        )
        reasoning = reasoning.split("</think>", 1)[0].strip()
        if force_final:
            prefix = "{\"done\": true, \"final_answer\": \""
        elif can_finish:
            prefix = "{\"done\": "
        else:
            prefix = "{\"done\": false, \"sub_question\": \""
        prefilled_messages = list(messages) + [
            {
                "role": "assistant",
                "reasoning_content": reasoning.strip(),
                "content": text_content(prefix),
            }
        ]
        inputs = self.processor.apply_chat_template(
            prefilled_messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
            **self.chat_template_kwargs,
        )
        inputs = inputs.to(self.model.device)
        visible_token_limit = min(128, max(64, self.max_new_tokens))
        generated = self._run_model_generate(
            inputs, max_new_tokens=visible_token_limit, do_sample=False
        )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated)
        ]
        continuation = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        combined = prefix + continuation
        objects = json_objects_from_text(combined)
        if not objects:
            objects = json_objects_from_text(continuation)
        chose_final = force_final
        evidence_list: list[str] = []
        value: str | None = None
        if objects:
            candidate = objects[-1]
            done_val = candidate.get("done")
            if isinstance(done_val, str):
                done_val = done_val.strip().lower() == "true"
            if not force_final and can_finish and isinstance(done_val, bool):
                chose_final = done_val
            key = "final_answer" if chose_final else "sub_question"
            alt_key = "answer" if chose_final else "question"
            raw_val = candidate.get(key) or candidate.get(alt_key)
            if isinstance(raw_val, str) and raw_val.strip():
                value = raw_val.strip()
            if chose_final and isinstance(candidate.get("evidence"), list):
                evidence_list = [
                    str(item).strip()
                    for item in candidate["evidence"]
                    if str(item).strip()
                ]
        if value is None:
            if not force_final and can_finish:
                chose_final = continuation.lstrip().lower().startswith("true")
            fragment = continuation.lstrip()
            target_key = "\"final_answer\"" if chose_final else "\"sub_question\""
            if target_key in fragment:
                fragment = fragment.split(target_key, 1)[1].lstrip()
                if fragment.startswith(":"):
                    fragment = fragment[1:].lstrip()
            if fragment.startswith("\""):
                fragment = fragment[1:]
            escaped = False
            end_index: int | None = None
            for index, character in enumerate(fragment):
                if character == "\"" and not escaped:
                    end_index = index
                    break
                if character == "\\" and not escaped:
                    escaped = True
                else:
                    escaped = False
            raw_value = fragment[:end_index] if end_index is not None else fragment
            raw_value = raw_value.strip().splitlines()[0] if raw_value.strip() else ""
            try:
                value = json.loads("\"" + raw_value + "\"") if raw_value else ""
            except json.JSONDecodeError:
                value = raw_value.replace("\\\"", "\"").strip()
        if not value:
            value = (
                "Unable to determine from the available evidence."
                if chose_final
                else "Is the relevant evidence visible in the selected frames?"
            )
        if not chose_final:
            normalized_value = normalize_model_sub_question(
                value, self.entity_genders
            )
            if normalized_value is None:
                print(
                    "[model] Discarded malformed or non-English sub-question output.",
                    file=sys.stderr, flush=True
                )
                value = safe_fallback_sub_question(())
            else:
                value = normalized_value
        payload: dict[str, Any] = (
            {"done": True, "final_answer": value}
            if chose_final
            else {"done": False, "sub_question": value}
        )
        if chose_final and evidence_list:
            payload["evidence"] = evidence_list
        return "</think>\n\n" + json.dumps(payload, ensure_ascii=False)


    def decide(
        self,
        messages: Sequence[dict[str, Any]],
        asked_questions: Sequence[str],
        min_subquestions: int,
        force_final: bool = False,
        answers: Sequence[str] = (),
    ) -> Decision:
        can_finish = (
            len(asked_questions) >= min_subquestions
            and ("Yes" in answers or not answers)
        )
        prefer_final_from_kg = (
            force_final
            or (min_subquestions <= 0 and self.same_question_in_kg and not asked_questions)
        )
        raw = self.generate(messages)
        if self.requires_complete_thinking and "</think>" not in raw:
            raw = self.complete_structured_output(
                messages, raw, prefer_final_from_kg, can_finish=can_finish
            )
        error: str | None
        try:
            decision = parse_decision(raw)
            if decision.sub_question:
                decision.sub_question = (
                    normalize_model_sub_question(
                        decision.sub_question, self.entity_genders
                    )
                    or decision.sub_question
                )
            error = validate_decision(
                decision,
                asked_questions,
                min_subquestions,
                force_final,
                self.original_question,
                self.reference_answer,
                entity_genders=self.entity_genders,
                prior_kg_questions=self.prior_kg_questions,
                added_relations=self.added_relations,
                answers=answers,
            )
        except ValueError as exc:
            decision = Decision(done=False)
            error = str(exc)
        if error is None:
            return decision

        repair_force_final = force_final or (
            can_finish and self.same_question_in_kg
        )
        repair_instruction = (
            f"Your last response was invalid: {error} "
            "Correct it now. Check the KG character genders and existing relations carefully. "
            "Any sub-question must use ASCII English, begin with an English binary auxiliary, "
            "respect character genders in the KG, never repeat a question/relation already in "
            "the KG, and contain at most 25 words. Output only one valid JSON object with no explanation."
        )
        repair_messages = list(messages) + [
            {"role": "assistant", "content": text_content(raw)},
            {"role": "user", "content": text_content(repair_instruction)},
        ]
        repaired_raw = self.generate(repair_messages)
        if self.requires_complete_thinking and "</think>" not in repaired_raw:
            repaired_raw = self.complete_structured_output(
                repair_messages,
                repaired_raw,
                repair_force_final,
                can_finish=can_finish,
            )
        try:
            repaired = parse_decision(repaired_raw)
        except ValueError:
            repaired_raw = self.complete_structured_output(
                repair_messages,
                repaired_raw,
                repair_force_final,
                can_finish=can_finish,
            )
            repaired = parse_decision(repaired_raw)
        if repaired.sub_question:
            repaired.sub_question = (
                normalize_model_sub_question(
                    repaired.sub_question, self.entity_genders
                )
                or repaired.sub_question
            )
        repaired_error = validate_decision(
            repaired,
            asked_questions,
            min_subquestions,
            force_final,
            self.original_question,
            self.reference_answer,
            entity_genders=self.entity_genders,
            prior_kg_questions=self.prior_kg_questions,
            added_relations=self.added_relations,
            answers=answers,
        )
        if repaired_error and not force_final:
            if can_finish and self.same_question_in_kg:
                final_raw = self.complete_structured_output(
                    repair_messages, repaired_raw, True, can_finish=True
                )
                final_decision = parse_decision(final_raw)
                if final_decision.done and final_decision.final_answer:
                    if not final_decision.evidence and self.added_relations:
                        final_decision.evidence = [
                            f"Verified in expanded KG: {rel.get('fact', '')}"
                            for rel in self.added_relations[:3]
                            if rel.get("fact")
                        ]
                    return final_decision

            normalized_question = normalize_model_sub_question(
                repaired.sub_question or "", self.entity_genders
            )
            if normalized_question:
                normalized_decision = Decision(
                    done=False, sub_question=normalized_question
                )
                normalized_error = validate_decision(
                    normalized_decision,
                    asked_questions,
                    min_subquestions,
                    force_final,
                    self.original_question,
                    self.reference_answer,
                    entity_genders=self.entity_genders,
                    prior_kg_questions=self.prior_kg_questions,
                    added_relations=self.added_relations,
                    answers=answers,
                )
                if normalized_error is None:
                    print(
                        "[model] Normalized the sub-question to yes/no form.",
                        file=sys.stderr, flush=True
                    )
                    return normalized_decision

            fallback_decision = Decision(
                done=False,
                sub_question=safe_fallback_sub_question(
                    list(asked_questions) + list(self.prior_kg_questions)
                ),
            )
            fallback_error = validate_decision(
                fallback_decision,
                asked_questions,
                min_subquestions,
                force_final,
                self.original_question,
                self.reference_answer,
                entity_genders=self.entity_genders,
                prior_kg_questions=self.prior_kg_questions,
                added_relations=self.added_relations,
                answers=answers,
            )
            if fallback_error is None:
                print(
                    "[model] Replaced an invalid sub-question with a safe yes/no "
                    "evidence question.",
                    file=sys.stderr, flush=True
                )
                return fallback_decision
        if repaired_error:
            raise RuntimeError(
                "The model failed the yes/no output constraint after one repair: "
                f"{repaired_error}"
            )
        return repaired


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write(chr(10))
    temporary_path.replace(path)


def build_type_subtype_map(kg_data: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    type_to_subtypes: dict[str, set[str]] = defaultdict(set)
    for episode in kg_data:
        episode_knowledge = episode.get("episode_knowledge", {})
        if not isinstance(episode_knowledge, dict):
            continue
        for head, head_info in episode_knowledge.items():
            if not is_valid_kg_entity_name(str(head)) or not isinstance(head_info, dict):
                continue
            relationships = head_info.get("relationships", {})
            if not isinstance(relationships, dict):
                continue
            for tail, relation_info in relationships.items():
                if not is_valid_kg_entity_name(str(tail)) or not isinstance(relation_info, dict):
                    continue
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
                if len(relation_types) != len(subtypes):
                    print(
                        f"[kg-expansion] Skipping mismatched type/subtype: "
                        f"{head} -> {tail}",
                        file=sys.stderr,
                    )
                    continue
                for relation_name, subtype_name in zip(relation_types, subtypes):
                    if isinstance(relation_name, str) and isinstance(subtype_name, str):
                        type_to_subtypes[relation_name].add(subtype_name)
    return {
        relation_name: sorted(subtypes)
        for relation_name, subtypes in sorted(type_to_subtypes.items())
    }


def save_transcript_csv(
    path: Path, transcript: Sequence[dict[str, str]], episode_id: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("level", "episode", "question", "answer")
        )
        writer.writeheader()
        for turn in transcript:
            writer.writerow(
                {
                    "level": "dialogue",
                    "episode": episode_id,
                    "question": turn.get("sub_question", ""),
                    "answer": turn.get("answer", ""),
                }
            )


def parse_relation_json(text: str) -> dict[str, Any]:
    objects = json_objects_from_text(text.strip())
    if not objects:
        raise ValueError(f"Relation JSON parsing failed. Raw output: {text}")
    return objects[-1]


def call_relation_json(
    agent: QwenSubQuestionAgent, prompt: str, max_new_tokens: int
) -> dict[str, Any]:
    raw_output = agent.generate_text(
        prompt, max_new_tokens=max_new_tokens, enable_thinking=False
    )
    try:
        return parse_relation_json(raw_output)
    except ValueError:
        repair_prompt = (
            prompt
            + chr(10) * 2
            + "Your prior response was invalid. Return only the requested JSON object."
        )
        repaired_output = agent.generate_text(
            repair_prompt, max_new_tokens=max_new_tokens, enable_thinking=False
        )
        return parse_relation_json(repaired_output)


def resolve_canonical_entity_name(
    raw_name: str,
    known_entities: Sequence[str],
    original_question: str = "",
    final_answer: str = "",
) -> str | None:
    cleaned = " ".join(str(raw_name).strip().split())
    if not cleaned:
        return None
    canonical_by_lower = {entity.lower(): entity for entity in known_entities if is_valid_kg_entity_name(entity)}
    for name in KNOWN_CHARACTER_GENDERS:
        canonical_by_lower.setdefault(name, name.capitalize())
    if cleaned.lower() in canonical_by_lower:
        return canonical_by_lower[cleaned.lower()]
    if is_valid_kg_entity_name(cleaned):
        return cleaned

    # If raw_name is an indirect phrase like "the woman whom Dokyung meets" or "the woman he meets",
    # resolve it using final_answer (excluding entities already named in original_question).
    question_lower = original_question.lower()
    answer_lower = final_answer.lower()
    answer_entities = [
        canon
        for lower_name, canon in canonical_by_lower.items()
        if re.search(rf"\b{re.escape(lower_name)}\b", answer_lower)
        and not re.search(rf"\b{re.escape(lower_name)}\b", question_lower)
    ]
    if len(answer_entities) == 1:
        return answer_entities[0]
    return None


def build_relation_type_prompt(
    question: str,
    answer: str,
    type_subtype_map: dict[str, list[str]],
    original_question: str = "",
    final_answer: str = "",
    known_entities: Sequence[str] = (),
) -> str:
    type_text = chr(10).join(f"- {name}" for name in type_subtype_map)
    entities_text = ", ".join(known_entities) if known_entities else "Dokyung, Haeyoung1, Haeyoung2, Sukyung, Jinsang, Taejin, Hun, Jiya"
    context_lines = []
    if original_question:
        context_lines.append(f"Original question: {original_question}")
    if final_answer:
        context_lines.append(f"Final answer to original question: {final_answer}")
    context_Lines_str = chr(10).join(context_lines)
    return f"""You extract explicit character relationships for a DramaQA knowledge graph.

{context_Lines_str}
Known episode character names:
{entities_text}

Sub-question:
{question}

Sub-answer:
{answer}

Existing Type candidates:
{type_text or "- none"}

Instructions:
1. Extract the character-to-character relationship asserted by the confirmed yes/no exchange.
2. Head and Tail MUST be canonical named characters (e.g., Dokyung, Haeyoung1, Haeyoung2), NEVER descriptive phrases like "the woman", "the woman he meets", or "the unnamed person".
3. If the sub-question uses an indirect reference (such as "the woman whom Dokyung meets", "the woman he meets", "the person", "he", "she"), resolve it to the canonical character name using the Original question and Final answer above.
4. Preserve direction Head -> Tail.
5. Prefer an exact existing Type string when semantically appropriate; otherwise create one concise broad Type (e.g., work, social, romantic, family).
6. If the sub-question is purely about an object, solo action, or location with no character-to-character relationship, return null for all fields.

Return only one JSON object:
{{"head": "...", "type": "...", "tail": "..."}}
or
{{"head": null, "type": null, "tail": null}}""".strip()


def select_relation_type(
    agent: QwenSubQuestionAgent,
    question: str,
    answer: str,
    type_subtype_map: dict[str, list[str]],
    original_question: str = "",
    final_answer: str = "",
    known_entities: Sequence[str] = (),
) -> dict[str, str] | None:
    result = call_relation_json(
        agent,
        build_relation_type_prompt(
            question,
            answer,
            type_subtype_map,
            original_question=original_question,
            final_answer=final_answer,
            known_entities=known_entities,
        ),
        max_new_tokens=128,
    )
    required_keys = {"head", "type", "tail"}
    if set(result) != required_keys:
        raise ValueError(f"Unexpected relation Type output: {result}")
    values = [result[key] for key in ("head", "type", "tail")]
    if all(value is None for value in values):
        return None
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"Incomplete relation Type output: {result}")
    head = resolve_canonical_entity_name(
        str(result["head"]).strip(),
        known_entities,
        original_question=original_question,
        final_answer=final_answer,
    )
    tail = resolve_canonical_entity_name(
        str(result["tail"]).strip(),
        known_entities,
        original_question=original_question,
        final_answer=final_answer,
    )
    rel_type = str(result["type"]).strip()
    if not head or not tail or head == tail or not rel_type:
        return None
    return {"head": head, "type": rel_type, "tail": tail}


def build_relation_subtype_prompt(
    question: str,
    answer: str,
    head: str,
    selected_type: str,
    tail: str,
    subtype_candidates: Sequence[str],
    original_question: str = "",
    final_answer: str = "",
) -> str:
    if subtype_candidates:
        candidate_text = chr(10).join(f"- {name}" for name in subtype_candidates)
    else:
        candidate_text = "- none; create one concise specific Subtype"
    context_block = ""
    if original_question or final_answer:
        context_block = (
            f"Original question: {original_question}\n"
            f"Final answer: {final_answer}\n\n"
        )
    return f"""Determine the specific Subtype of an explicit DramaQA character relationship.

{context_block}Sub-question:
{question}

Sub-answer:
{answer}

Head: {head}
Tail: {tail}
Selected Type: {selected_type}

Existing Subtype candidates:
{candidate_text}

In DramaQA KG, the Subtype under Head -> Tail describes what Tail is to Head (and respects Tail's gender). Prefer an exact existing candidate.
Only create a new concise Subtype if no candidate fits.

Return only one JSON object:
{{"subtype": "..."}}""".strip()


def select_relation_subtype(
    agent: QwenSubQuestionAgent,
    question: str,
    answer: str,
    head: str,
    selected_type: str,
    tail: str,
    type_subtype_map: dict[str, list[str]],
    original_question: str = "",
    final_answer: str = "",
) -> tuple[str, list[str]]:
    candidates = list(type_subtype_map.get(selected_type, []))
    result = call_relation_json(
        agent,
        build_relation_subtype_prompt(
            question,
            answer,
            head,
            selected_type,
            tail,
            candidates,
            original_question=original_question,
            final_answer=final_answer,
        ),
        max_new_tokens=64,
    )
    subtype = result.get("subtype")
    if not isinstance(subtype, str) or not subtype.strip():
        raise ValueError(f"Unexpected relation Subtype output: {result}")
    return subtype.strip(), candidates


def extract_kg_quadruple(
    agent: QwenSubQuestionAgent,
    question: str,
    answer: str,
    episode_id: str,
    type_subtype_map: dict[str, list[str]],
    original_question: str = "",
    final_answer: str = "",
    known_entities: Sequence[str] = (),
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    if normalize_binary_answer(answer) != "Yes":
        return None, {"reason": "negative_answer"}
    type_result = select_relation_type(
        agent,
        question,
        answer,
        type_subtype_map,
        original_question=original_question,
        final_answer=final_answer,
        known_entities=known_entities,
    )
    if type_result is None:
        return None, {"reason": "no_explicit_character_relation"}
    head = type_result["head"]
    selected_type = type_result["type"]
    tail = type_result["tail"]
    selected_subtype, candidates = select_relation_subtype(
        agent,
        question,
        answer,
        head,
        selected_type,
        tail,
        type_subtype_map,
        original_question=original_question,
        final_answer=final_answer,
    )
    result = {
        "episode": episode_id,
        "head": head,
        "type": selected_type,
        "subtype": selected_subtype,
        "tail": tail,
    }
    info = {
        "type_is_existing": selected_type in type_subtype_map,
        "subtype_candidates": candidates,
        "subtype_is_existing": selected_subtype in candidates,
    }
    return result, info

