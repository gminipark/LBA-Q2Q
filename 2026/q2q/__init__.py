"""Modular DramaQA Q2Q and Knowledge Graph expansion package."""

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
