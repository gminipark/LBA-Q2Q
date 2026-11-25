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
from q2q.visualization import (
    build_type_subtype_map,
    save_transcript_csv,
    kg_graph_signature,
    write_kg_visualization_html,
)
from q2q.kg_expansion import (
    parse_relation_json,
    call_relation_json,
    resolve_canonical_entity_name,
    build_relation_type_prompt,
    select_relation_type,
    build_relation_subtype_prompt,
    select_relation_subtype,
    extract_kg_quadruple,
    ensure_episode_entity,
    add_quadruple_to_episode,
    find_target_episode,
    expand_kg_from_dialogue,
)
from q2q.dialogue import (
    read_binary_answer,
    run_dialogue,
    print_sample,
)


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
