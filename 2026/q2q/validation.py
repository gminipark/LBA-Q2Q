"""Sub-question normalization, JSON decision parsing, and KG validation guards."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from q2q.config import (
    ENGLISH_BINARY_PREFIXES,
    ENGLISH_WH_PREFIXES,
    KNOWN_CHARACTER_GENDERS,
    TOKEN_RE,
)
from q2q.schemas import Decision, meaningful_tokens, normalize_question, tokenize


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

