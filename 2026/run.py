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

