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

from q2q.config import (
    SCRIPT_DIR,
    DEFAULT_DATASET_ROOT,
    DEFAULT_KG_PATH,
    DEFAULT_QA_DIR,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TYPE_SUBTYPE_MAP_PATH,
    DEFAULT_TRANSCRIPT_CSV_PATH,
    DEFAULT_EXPANDED_KG_PATH,
    DEFAULT_EXPANSION_LOG_PATH,
    DEFAULT_VISUALIZATION_PATH,
    SAMPLE_QUESTION,
    TOKEN_RE,
    IMAGE_SUFFIXES,
    ENGLISH_WH_PREFIXES,
    ENGLISH_BINARY_PREFIXES,
    KNOWN_CHARACTER_GENDERS,
    MALE_RELATION_SUBTYPES,
    FEMALE_RELATION_SUBTYPES,
)
from q2q.schemas import (
    QASample,
    SceneRecord,
    Decision,
    GenerationCancelled,
    load_json,
    write_json_atomic,
    tokenize,
    meaningful_tokens,
    normalize_question,
    normalize_binary_answer,
    unique_in_order,
    episode_key_from_id,
    format_gender_label,
    relation_pairs,
)
from q2q.kb import (
    is_valid_kg_entity_name,
    sanitize_kg_episodes,
    DramaQAKnowledgeBase,
)
from q2q.frames import (
    resolve_image_root,
    natural_sort_key,
    evenly_spaced,
    representative_frames,
)
from q2q.validation import (
    is_plausible_english_question,
    is_yes_no_question,
    rewrite_ambiguous_entity_clauses,
    normalize_model_sub_question,
    safe_fallback_sub_question,
    json_objects_from_text,
    parse_decision,
    kg_entity_consistency_reason,
    already_known_in_kg_reason,
    direct_answer_probe_reason,
    validate_decision,
)
from q2q.prompts import (
    build_turn_followup_prompt,
    text_content,
    system_prompt,
    initial_messages,
)
from q2q.agent import (
    QwenSubQuestionAgent,
)


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



def kg_graph_signature(
    kg_data: Sequence[dict[str, Any]],
) -> dict[str, list[str]]:
    nodes: set[str] = set()
    edges: set[str] = set()
    for episode in kg_data:
        episode_id = str(episode.get("episode", ""))
        knowledge = episode.get("episode_knowledge", {})
        if not isinstance(knowledge, dict):
            continue
        for head, info in knowledge.items():
            head_name = str(head)
            nodes.add(head_name)
            if not isinstance(info, dict):
                continue
            relationships = info.get("relationships", {})
            if not isinstance(relationships, dict):
                continue
            for tail, relation_info in relationships.items():
                tail_name = str(tail)
                nodes.add(tail_name)
                if not isinstance(relation_info, dict):
                    continue
                for relation_type, subtype in relation_pairs(relation_info):
                    edges.add(
                        "|".join(
                            (episode_id, head_name, relation_type, subtype, tail_name)
                        )
                    )
    return {"nodes": sorted(nodes), "edges": sorted(edges)}



def write_kg_visualization_html(
    kg_path: Path,
    output_path: Path,
    baseline_signature: dict[str, list[str]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relative_path = os.path.relpath(
        kg_path, start=output_path.parent
    ).replace(os.sep, "/")
    baseline_signature = baseline_signature or {"nodes": [], "edges": []}
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DramaQA KG Comparison</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
:root{--ink:#17211b;--muted:#69746c;--line:#d8d7cc;--green:#174f3d;--orange:#ef6f3c;--paper:#faf9f4}
*{box-sizing:border-box}body{font-family:Inter,Arial,sans-serif;margin:0;color:var(--ink);background:var(--paper)}
#toolbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:12px 14px;border-bottom:1px solid var(--line);background:white}
#info{font-size:13px;font-weight:700;margin-right:auto}.legend{display:flex;gap:12px;font-size:12px;color:var(--muted)}
.key{display:inline-flex;align-items:center;gap:5px}.line{width:24px;height:3px;border-radius:3px;background:#9ba59e}.line.new{height:5px;background:var(--orange)}
button{border:1px solid var(--line);background:white;border-radius:9px;padding:7px 10px;font-weight:700;color:var(--ink);cursor:pointer}
button.active{background:var(--green);border-color:var(--green);color:white}#network{height:820px;background:white}
</style></head><body>
<div id="toolbar"><div id="info">Loading KG comparison...</div><div class="legend"><span class="key"><i class="line"></i>기존 edge</span><span class="key"><i class="line new"></i>이번에 추가</span></div><button id="showAll" class="active">전체</button><button id="showAdded">추가 edge만</button><button id="relayout">레이아웃 다시 계산</button></div>
<div id="network"></div>
<script>
const KG_URL=__KG_URL__; const baselineNodeIds=new Set(__BASELINE_NODES__); const baselineEdgeIds=new Set(__BASELINE_EDGES__);
const nodes=new vis.DataSet([]); const edges=new vis.DataSet([]);
const network=new vis.Network(document.getElementById("network"),{nodes:nodes,edges:edges},{
  edges:{arrows:{to:{enabled:true}},font:{align:"middle",size:11}},
  nodes:{shape:"dot",font:{size:13}},
  physics:{enabled:true,solver:"forceAtlas2Based",stabilization:{enabled:true,iterations:450}},
  interaction:{hover:true,navigationButtons:true,keyboard:true}
});
let allNodes=[]; let allEdges=[]; let mode="all"; let lastPayload=null;
let freezeTimer=null;
function asArray(value){return Array.isArray(value)?value:(value===undefined||value===null?[]:[value]);}
function collect(data){
  const nodeIds=new Set(); const edgeMap=new Map();
  for(const episode of data){
    const episodeId=String(episode.episode||""); const knowledge=episode.episode_knowledge||{};
    for(const [head,info] of Object.entries(knowledge)){
      nodeIds.add(head);
      for(const [tail,rel] of Object.entries(info.relationships||{})){
        nodeIds.add(tail); const types=asArray(rel.type); const subtypes=asArray(rel.subtype);
        for(let i=0;i<Math.min(types.length,subtypes.length);i++){
          const id=[episodeId,head,types[i],subtypes[i],tail].join("|");
          edgeMap.set(id,{id:id,from:head,to:tail,label:String(subtypes[i]),relationType:String(types[i]),episode:episodeId});
        }
      }
    }
  }
  return {nodeIds:nodeIds,edgeMap:edgeMap};
}
function freezeLayout(){
  clearTimeout(freezeTimer);
  network.setOptions({physics:{enabled:false}});
  network.fit({animation:{duration:250,easingFunction:"easeInOutQuad"}});
}
function settleAndFreeze(){
  clearTimeout(freezeTimer);
  network.setOptions({physics:{enabled:true,solver:"forceAtlas2Based",stabilization:{enabled:true,iterations:450}}});
  network.stabilize(450);
  freezeTimer=setTimeout(freezeLayout,2000);
}
function applyMode(){
  const visibleEdges=mode==="added"?allEdges.filter((edge)=>edge.added):allEdges;
  const connected=new Set(); visibleEdges.forEach((edge)=>{connected.add(edge.from);connected.add(edge.to);});
  const visibleNodes=mode==="added"?allNodes.filter((node)=>connected.has(node.id)):allNodes;
  nodes.clear();edges.clear();nodes.add(visibleNodes);edges.add(visibleEdges);settleAndFreeze();
  document.getElementById("showAll").classList.toggle("active",mode==="all");
  document.getElementById("showAdded").classList.toggle("active",mode==="added");
}
function buildComparison(currentData){
  const current=collect(currentData);
  allNodes=Array.from(current.nodeIds).map((id)=>{
    const added=!baselineNodeIds.has(id); return {id:id,label:id,title:(added?"이번에 추가된 인물: ":"Character: ")+id,color:added?{background:"#ffd7c7",border:"#ef6f3c"}:{background:"#dcebdc",border:"#4f7961"}};
  });
  allEdges=Array.from(current.edgeMap.values()).map((edge)=>{
    const added=!baselineEdgeIds.has(edge.id); return {...edge,added:added,title:(added?"[이번에 추가] ":"[기존] ")+"Episode: "+edge.episode+" | Type: "+edge.relationType,color:{color:added?"#ef6f3c":"#9ba59e",highlight:added?"#d94f1d":"#57645c"},width:added?5:1.2,arrows:"to"};
  });
  const addedCount=allEdges.filter((edge)=>edge.added).length;
  document.getElementById("info").textContent="Nodes: "+allNodes.length+" | 기존 edges: "+baselineEdgeIds.size+" | 현재 edges: "+allEdges.length+" | 이번에 추가: "+addedCount;
  applyMode();
}
async function fetchJson(url){if(!url)return [];const response=await fetch(url+"?timestamp="+Date.now());if(!response.ok)throw new Error("HTTP "+response.status+" for "+url);return response.json();}
async function loadGraph(){
  try{
    const currentData=await fetchJson(KG_URL);
    const payload=JSON.stringify(currentData);if(payload===lastPayload)return;lastPayload=payload;buildComparison(currentData);
  }catch(error){document.getElementById("info").textContent="Error loading KG: "+error.message;}
}
document.getElementById("showAll").onclick=()=>{mode="all";applyMode();};
document.getElementById("showAdded").onclick=()=>{mode="added";applyMode();};
document.getElementById("relayout").onclick=settleAndFreeze;
loadGraph();setInterval(loadGraph,5000);
</script></body></html>
    """.replace("__KG_URL__", json.dumps(relative_path)).replace(
        "__BASELINE_NODES__", json.dumps(baseline_signature["nodes"])
    ).replace(
        "__BASELINE_EDGES__", json.dumps(baseline_signature["edges"])
    )
    output_path.write_text(html, encoding="utf-8")



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



def read_binary_answer(prefilled: list[str]) -> str | None:
    while True:
        if prefilled:
            raw = prefilled.pop(0)
            print(f"Your answer [yes/no]: {raw}")
        else:
            try:
                raw = input("Your answer [yes/no, quit]: ")
            except EOFError:
                return None
        if raw.strip().lower() in {"quit", "exit", "q"}:
            return None
        answer = normalize_binary_answer(raw)
        if answer:
            return answer
        print("Please answer only yes/no.")



def run_dialogue(
    agent: QwenSubQuestionAgent,
    messages: list[dict[str, Any]],
    min_subquestions: int,
    max_subquestions: int,
    prefilled_answers: list[str],
) -> tuple[list[dict[str, str]], Decision | None]:
    transcript: list[dict[str, str]] = []
    asked_questions: list[str] = []

    for turn in range(1, max_subquestions + 1):
        answers_so_far = [item["answer"] for item in transcript]
        decision = agent.decide(
            messages,
            asked_questions=asked_questions,
            min_subquestions=min_subquestions,
            answers=answers_so_far,
        )
        if decision.done:
            print(f"\nFinal answer: {decision.final_answer}")
            if decision.evidence:
                print("Evidence:")
                for item in decision.evidence:
                    print(f"  - {item}")
            return transcript, decision

        assert decision.sub_question is not None
        question = decision.sub_question
        print(f"\nSub-question {turn}: {question}")
        answer = read_binary_answer(prefilled_answers)
        if answer is None:
            print("Dialogue ended without a final answer.")
            return transcript, None

        asked_questions.append(question)
        transcript.append({"sub_question": question, "answer": answer})
        answers_so_far = [item["answer"] for item in transcript]
        messages.append(
            {
                "role": "assistant",
                "content": text_content(
                    json.dumps(
                        {"done": False, "sub_question": question},
                        ensure_ascii=False,
                    )
                ),
            }
        )
        remaining = max_subquestions - turn
        messages.append(
            {
                "role": "user",
                "content": text_content(
                    build_turn_followup_prompt(
                        answer=answer,
                        remaining=remaining,
                        asked_count=len(asked_questions),
                        min_subquestions=min_subquestions,
                        answers_so_far=answers_so_far,
                    )
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": text_content(
                "The sub-question limit has been reached. Do not ask another "
                "question. Return done=true with the final answer and concise evidence."
            ),
        }
    )
    final_decision = agent.decide(
        messages,
        asked_questions=asked_questions,
        min_subquestions=min_subquestions,
        force_final=True,
        answers=[item["answer"] for item in transcript],
    )
    print(f"\nFinal answer: {final_decision.final_answer}")
    if final_decision.evidence:
        print("Evidence:")
        for item in final_decision.evidence:
            print(f"  - {item}")
    return transcript, final_decision



def print_sample(kb: DramaQAKnowledgeBase, image_root: Path, max_images: int) -> None:
    sample = kb.find_exact_qa(SAMPLE_QUESTION)
    if sample is None:
        raise RuntimeError("The built-in sample was not found in the KG.")
    records, _ = kb.retrieve(sample.question, top_k=3)
    images = representative_frames(image_root, records, max_images=max_images)
    print("Dataset sample")
    print(f"  Question: {sample.question}")
    print(f"  Reference answer: {sample.answer}")
    print(f"  Relevant scene(s): {', '.join(sample.scene_ids)}")
    if images:
        print(f"  Representative image: {images[0][1]}")
    print("\nIllustrative dialogue (actual model wording can differ)")
    print("  Agent: Does the woman whom Dokyung meets work at the same company as Haeyoung1?")
    print("  User: Yes")
    print("  Final answer: Haeyoung2")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive yes/no sub-questioning over DramaQA images and KG."
    )
    parser.add_argument("--question", help="Original DramaQA question.")
    parser.add_argument(
        "--vid",
        help="Comma-separated scene IDs. Overrides automatic QA/KG retrieval.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run the built-in episode-16 sample through the model.",
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Show one real dataset sample and an illustrative dialogue without loading the model.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--kg-path", type=Path)
    parser.add_argument("--qa-dir", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
    )
    parser.add_argument("--top-scenes", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--min-subquestions", type=int, default=1)
    parser.add_argument("--max-subquestions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--answers",
        help="Optional comma-separated yes/no answers for a non-interactive demo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Retrieve KG/images but do not load or run the Qwen model.",
    )
    parser.add_argument(
        "--show-context", action="store_true", help="Print the retrieved KG JSON."
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download model files from Hugging Face.",
    )
    parser.add_argument(
        "--no-kg-expansion",
        action="store_true",
        help="Skip relation extraction and KG expansion after the dialogue.",
    )
    parser.add_argument(
        "--expanded-kg-path", type=Path, default=DEFAULT_EXPANDED_KG_PATH
    )
    parser.add_argument(
        "--type-subtype-map-path",
        type=Path,
        default=DEFAULT_TYPE_SUBTYPE_MAP_PATH,
    )
    parser.add_argument(
        "--transcript-csv-path", type=Path, default=DEFAULT_TRANSCRIPT_CSV_PATH
    )
    parser.add_argument(
        "--expansion-log-path", type=Path, default=DEFAULT_EXPANSION_LOG_PATH
    )
    parser.add_argument(
        "--kg-visualization-path",
        type=Path,
        default=DEFAULT_VISUALIZATION_PATH,
    )
    parser.add_argument("--output", type=Path, help="Optional path for dialogue JSON.")
    return parser



def validate_args(args: argparse.Namespace) -> None:
    if args.top_scenes < 1:
        raise ValueError("--top-scenes must be at least 1.")
    if args.max_images < 0:
        raise ValueError("--max-images cannot be negative.")
    if args.min_subquestions < 0:
        raise ValueError("--min-subquestions cannot be negative.")
    if args.max_subquestions < 1:
        raise ValueError("--max-subquestions must be at least 1.")
    if args.min_subquestions > args.max_subquestions:
        raise ValueError("--min-subquestions cannot exceed --max-subquestions.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")
    if args.top_k < 0:
        raise ValueError("--top-k cannot be negative.")



def compute_effective_min_subquestions(
    requested_min: int, prior_knowledge: dict[str, Any]
) -> int:
    if prior_knowledge.get("same_question_already_in_kg"):
        return 0
    same_q_turns = sum(
        1
        for item in prior_knowledge.get("verified_subquestions", [])
        if isinstance(item, dict) and item.get("from_same_question")
    )
    if same_q_turns > 0:
        return max(0, requested_min - same_q_turns)
    return requested_min



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        dataset_root = args.dataset_root.expanduser().resolve()
        kg_path = (
            args.kg_path.expanduser().resolve()
            if args.kg_path
            else dataset_root / "KG" / "AnotherMissOhQA_test_episode_scene_kg.json"
        )
        qa_dir = (
            args.qa_dir.expanduser().resolve()
            if args.qa_dir
            else dataset_root / "QA"
        )
        expanded_kg_path = args.expanded_kg_path.expanduser().resolve()
        expansion_log_path = args.expansion_log_path.expanduser().resolve()
        image_root = resolve_image_root(args.image_root)
        kb = DramaQAKnowledgeBase(
            kg_path=kg_path,
            qa_dir=qa_dir,
            expanded_kg_path=expanded_kg_path,
            expansion_log_path=expansion_log_path,
        )

        if args.show_sample:
            print_sample(kb, image_root, max_images=max(args.max_images, 1))
            return 0

        if args.sample:
            question = SAMPLE_QUESTION
        elif args.question:
            question = args.question.strip()
        else:
            question = input("Original question: ").strip()
        if not question:
            raise ValueError("The original question cannot be empty.")

        explicit_ids = (
            [value.strip() for value in args.vid.split(",") if value.strip()]
            if args.vid
            else None
        )
        records, retrieval_method = kb.retrieve(
            question,
            top_k=args.top_scenes,
            explicit_scene_ids=explicit_ids,
        )
        if not records:
            raise RuntimeError(
                "No relevant scene was found. Supply a scene ID with --vid."
            )

        matched_qa = kb.find_exact_qa(question)
        shot_ranges: dict[str, tuple[int, int]] = {}
        if matched_qa and matched_qa.shot_range:
            for scene_id in matched_qa.scene_ids:
                shot_ranges[scene_id] = matched_qa.shot_range
        images = representative_frames(
            image_root,
            records,
            max_images=args.max_images,
            shot_ranges=shot_ranges,
        )
        selected_shot_ids = {
            f"{'_'.join(scene_id.split('_')[:2])}_{path.parent.name}"
            for scene_id, path in images
        }
        prior_knowledge = kb.get_prior_verified_knowledge(question, records)
        effective_min_subquestions = compute_effective_min_subquestions(
            args.min_subquestions, prior_knowledge
        )
        context = kb.build_context(question, records, selected_shot_ids)

        print(f"[data] Retrieval: {retrieval_method}")
        print(f"[data] Scene(s): {', '.join(record.scene_id for record in records)}")
        print(f"[data] Representative frame(s): {len(images)}")
        for _, path in images:
            print(f"  - {path}")
        if prior_knowledge["same_question_already_in_kg"] or prior_knowledge["verified_subquestions"]:
            print(
                f"[data] Existing KG verification found "
                f"(verified_subquestions={len(prior_knowledge['verified_subquestions'])}, "
                f"expanded_relations={len(prior_knowledge['added_relations'])}, "
                f"effective_min_subquestions={effective_min_subquestions})"
            )
        missing_kg = [
            record.scene_id
            for record in records
            if not record.knowledge and record.episode_key not in kb.episodes
        ]
        if missing_kg:
            print(
                "[warning] The provided KG has no scene data for: "
                + ", ".join(missing_kg),
                file=sys.stderr,
            )
        if args.show_context:
            print(f"\nRetrieved KG context:\n{context}")
        if args.dry_run:
            print("\n[dry-run] Model loading and dialogue were skipped.")
            return 0

        messages = initial_messages(
            question,
            context,
            images,
            effective_min_subquestions,
            args.max_subquestions,
            prior_knowledge=prior_knowledge,
        )
        prior_kg_questions = [
            str(item["sub_question"])
            for item in prior_knowledge.get("verified_subquestions", [])
            if isinstance(item, dict) and item.get("sub_question")
        ]
        agent = QwenSubQuestionAgent(
            model_name=args.model,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            attn_implementation=args.attn_implementation,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            original_question=question,
            reference_answer=matched_qa.answer if matched_qa else None,
            entity_genders=kb.entity_genders,
            prior_kg_questions=prior_kg_questions,
            added_relations=prior_knowledge.get("added_relations", []),
            same_question_in_kg=bool(prior_knowledge.get("same_question_already_in_kg")),
        )
        prefilled_answers = (
            [value.strip() for value in args.answers.split(",") if value.strip()]
            if args.answers
            else []
        )
        transcript, final_decision = run_dialogue(
            agent,
            messages,
            min_subquestions=effective_min_subquestions,
            max_subquestions=args.max_subquestions,
            prefilled_answers=prefilled_answers,
        )

        kg_expansion_results: list[dict[str, Any]] = []
        kg_expansion_outputs: dict[str, str] | None = None
        if (
            final_decision is not None
            and transcript
            and not args.no_kg_expansion
        ):
            episode_keys = unique_in_order(
                record.episode_key for record in records
            )
            if len(episode_keys) > 1:
                print(
                    "[kg-expansion] Multiple episodes retrieved; using "
                    f"{episode_keys[0]}.",
                    file=sys.stderr,
                )
            episode_data = kb.episodes.get(episode_keys[0])
            if episode_data is None:
                raise ValueError(
                    f"Episode metadata not found: {episode_keys[0]}"
                )
            episode_id = str(episode_data.get("episode", episode_keys[0]))
            type_subtype_map_path = (
                args.type_subtype_map_path.expanduser().resolve()
            )
            transcript_csv_path = (
                args.transcript_csv_path.expanduser().resolve()
            )
            visualization_path = (
                args.kg_visualization_path.expanduser().resolve()
            )
            kg_expansion_results = expand_kg_from_dialogue(
                agent=agent,
                source_kg_path=kg_path,
                expanded_kg_path=expanded_kg_path,
                type_subtype_map_path=type_subtype_map_path,
                transcript_csv_path=transcript_csv_path,
                expansion_log_path=expansion_log_path,
                visualization_path=visualization_path,
                transcript=transcript,
                episode_id=episode_id,
                original_question=question,
                final_answer=(
                    final_decision.final_answer
                    or (matched_qa.answer if matched_qa else "")
                    or ""
                ),
            )
            status_counts = Counter(
                item["status"] for item in kg_expansion_results
            )
            print(
                "[kg-expansion] "
                + ", ".join(
                    f"{status}={count}"
                    for status, count in sorted(status_counts.items())
                )
            )
            print(f"[kg-expansion] Expanded KG: {expanded_kg_path}")
            kg_expansion_outputs = {
                "expanded_kg": str(expanded_kg_path),
                "type_subtype_map": str(type_subtype_map_path),
                "transcript_csv": str(transcript_csv_path),
                "expansion_log": str(expansion_log_path),
                "visualization": str(visualization_path),
            }


        if args.output:
            output_path = args.output.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "question": question,
                "retrieval_method": retrieval_method,
                "scene_ids": [record.scene_id for record in records],
                "images": [str(path) for _, path in images],
                "dialogue": transcript,
                "final": asdict(final_decision) if final_decision else None,
                "kg_expansion": kg_expansion_results,
                "kg_expansion_outputs": kg_expansion_outputs,
            }
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(f"[output] Saved dialogue to {output_path}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
