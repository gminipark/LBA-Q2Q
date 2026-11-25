"""Dialogue-driven relation extraction and episode KG expansion pipeline."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from threading import Event
from typing import Any, Sequence

from q2q.agent import QwenSubQuestionAgent
from q2q.config import KNOWN_CHARACTER_GENDERS
from q2q.kb import is_valid_kg_entity_name, sanitize_kg_episodes
from q2q.schemas import (
    GenerationCancelled,
    episode_key_from_id,
    load_json,
    normalize_binary_answer,
    normalize_question,
    relation_pairs,
)
from q2q.validation import json_objects_from_text
from q2q.visualization import (
    build_type_subtype_map,
    kg_graph_signature,
    save_transcript_csv,
    write_json_atomic,
    write_kg_visualization_html,
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



def ensure_episode_entity(
    episode_knowledge: dict[str, Any], entity: str
) -> None:
    if entity not in episode_knowledge or not isinstance(
        episode_knowledge.get(entity), dict
    ):
        episode_knowledge[entity] = {"events": {}, "relationships": {}}
    episode_knowledge[entity].setdefault("events", {})
    episode_knowledge[entity].setdefault("relationships", {})



def add_quadruple_to_episode(
    episode: dict[str, Any],
    head: str,
    relation_type: str,
    subtype: str,
    tail: str,
) -> str:
    episode_knowledge = episode.setdefault("episode_knowledge", {})
    if not isinstance(episode_knowledge, dict):
        raise ValueError("episode_knowledge must be a JSON object.")
    ensure_episode_entity(episode_knowledge, head)
    ensure_episode_entity(episode_knowledge, tail)
    relationships = episode_knowledge[head]["relationships"]
    if not isinstance(relationships, dict):
        relationships = {}
        episode_knowledge[head]["relationships"] = relationships
    new_pair = (relation_type, subtype)
    if tail not in relationships or not isinstance(relationships[tail], dict):
        relationships[tail] = {"type": relation_type, "subtype": subtype}
        return "added_new_relation"
    existing_pairs = relation_pairs(relationships[tail])
    if new_pair in existing_pairs:
        return "already_exists"
    existing_pairs.append(new_pair)
    relationships[tail] = {
        "type": [pair[0] for pair in existing_pairs],
        "subtype": [pair[1] for pair in existing_pairs],
    }
    return "added_additional_relation"



def find_target_episode(
    kg_data: Sequence[dict[str, Any]], episode_id: str
) -> dict[str, Any]:
    requested_key = episode_key_from_id(episode_id)
    for episode in kg_data:
        stored_id = str(episode.get("episode", ""))
        if stored_id == episode_id or episode_key_from_id(stored_id) == requested_key:
            return episode
    raise ValueError(f"Episode not found for KG expansion: {episode_id}")



def expand_kg_from_dialogue(
    agent: QwenSubQuestionAgent,
    source_kg_path: Path,
    expanded_kg_path: Path,
    type_subtype_map_path: Path,
    transcript_csv_path: Path,
    expansion_log_path: Path,
    visualization_path: Path,
    transcript: Sequence[dict[str, str]],
    episode_id: str,
    original_question: str = "",
    final_answer: str = "",
) -> list[dict[str, Any]]:
    working_source = (
        expanded_kg_path if expanded_kg_path.exists() else source_kg_path
    )
    base_kg_data = load_json(source_kg_path)
    if not isinstance(base_kg_data, list):
        raise ValueError(f"KG root must be a list: {source_kg_path}")
    kg_data = load_json(working_source)
    if not isinstance(kg_data, list):
        raise ValueError(f"KG root must be a list: {working_source}")
    sanitize_kg_episodes(kg_data)
    baseline_signature = kg_graph_signature(base_kg_data)
    type_subtype_map = build_type_subtype_map(kg_data)
    target_episode = find_target_episode(kg_data, episode_id)
    known_entities_set: set[str] = set()
    ep_knowledge = target_episode.get("episode_knowledge", {})
    if isinstance(ep_knowledge, dict):
        for ent in ep_knowledge:
            if is_valid_kg_entity_name(str(ent)):
                known_entities_set.add(str(ent))
    for scene_wrapper in target_episode.get("scene_level", []):
        if isinstance(scene_wrapper, dict):
            for payload in scene_wrapper.values():
                if isinstance(payload, dict) and isinstance(payload.get("scene_knowledge"), dict):
                    for ent in payload["scene_knowledge"]:
                        if is_valid_kg_entity_name(str(ent)):
                            known_entities_set.add(str(ent))
    known_entities = sorted(known_entities_set)

    effective_question = original_question or getattr(agent, "original_question", "") or ""
    effective_final_answer = (
        final_answer or getattr(agent, "reference_answer", "") or ""
    )
    results: list[dict[str, Any]] = []
    history_turns: list[dict[str, Any]] = []
    save_transcript_csv(transcript_csv_path, transcript, episode_id)

    for turn in transcript:
        question = str(turn.get("sub_question", "")).strip()
        answer = str(turn.get("answer", "")).strip()
        if not question or not answer:
            continue
        try:
            quadruple, selection_info = extract_kg_quadruple(
                agent,
                question,
                answer,
                episode_id,
                type_subtype_map,
                original_question=effective_question,
                final_answer=effective_final_answer,
                known_entities=known_entities,
            )
            if quadruple is None:
                results.append(
                    {
                        "question": question,
                        "answer": answer,
                        "status": "skipped",
                        "selection_info": selection_info,
                    }
                )
                history_turns.append(
                    {
                        "sub_question": question,
                        "answer": answer,
                        "quadruple": None,
                    }
                )
                continue
            status = add_quadruple_to_episode(
                target_episode,
                quadruple["head"],
                quadruple["type"],
                quadruple["subtype"],
                quadruple["tail"],
            )
            subtypes = type_subtype_map.setdefault(quadruple["type"], [])
            if quadruple["subtype"] not in subtypes:
                subtypes.append(quadruple["subtype"])
                subtypes.sort()
            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "status": status,
                    "quadruple": quadruple,
                    "selection_info": selection_info,
                }
            )
            history_turns.append(
                {
                    "sub_question": question,
                    "answer": answer,
                    "quadruple": quadruple,
                }
            )
        except GenerationCancelled:
            raise
        except (RuntimeError, ValueError) as exc:
            print(
                f"[kg-expansion] Failed for sub-question: {exc}",
                file=sys.stderr,
            )
            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "status": "error",
                    "error": str(exc),
                }
            )
            history_turns.append(
                {
                    "sub_question": question,
                    "answer": answer,
                    "quadruple": None,
                }
            )

    if history_turns:
        history = target_episode.setdefault("expanded_dialogue_history", [])
        if not isinstance(history, list):
            history = []
            target_episode["expanded_dialogue_history"] = history
        norm_orig = normalize_question(effective_question)
        updated_existing = False
        for existing in history:
            if (
                isinstance(existing, dict)
                and norm_orig
                and normalize_question(str(existing.get("original_question", ""))) == norm_orig
            ):
                existing_turns = existing.setdefault("turns", [])
                if isinstance(existing_turns, list):
                    seen_qs = {
                        normalize_question(str(t.get("sub_question", "")))
                        for t in existing_turns
                        if isinstance(t, dict)
                    }
                    for new_t in history_turns:
                        if normalize_question(new_t["sub_question"]) not in seen_qs:
                            existing_turns.append(new_t)
                if effective_final_answer:
                    existing["final_answer"] = effective_final_answer
                updated_existing = True
                break
        if not updated_existing:
            history.append(
                {
                    "original_question": effective_question,
                    "final_answer": effective_final_answer,
                    "episode": episode_id,
                    "turns": history_turns,
                }
            )

    stop_event = getattr(agent, "stop_event", None)
    if stop_event is not None and stop_event.is_set():
        raise GenerationCancelled("KG expansion was cancelled by the user.")
    write_json_atomic(expanded_kg_path, kg_data)
    write_json_atomic(type_subtype_map_path, type_subtype_map)
    combined_log: list[dict[str, Any]] = []
    if expansion_log_path.exists():
        try:
            existing_log = load_json(expansion_log_path)
            if isinstance(existing_log, list):
                combined_log.extend(
                    item for item in existing_log if isinstance(item, dict)
                )
        except Exception:
            pass
    existing_keys = {
        (
            normalize_question(str(item.get("question", ""))),
            json.dumps(item.get("quadruple"), sort_keys=True, ensure_ascii=False),
        )
        for item in combined_log
    }
    for item in results:
        key = (
            normalize_question(str(item.get("question", ""))),
            json.dumps(item.get("quadruple"), sort_keys=True, ensure_ascii=False),
        )
        if key not in existing_keys:
            combined_log.append(item)
            existing_keys.add(key)
    write_json_atomic(expansion_log_path, combined_log)
    write_kg_visualization_html(
        expanded_kg_path, visualization_path, baseline_signature
    )
    return results

