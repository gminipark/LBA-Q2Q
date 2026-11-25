"""System and multi-turn prompt builders for binary sub-questioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


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

