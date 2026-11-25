"""Path defaults, constants, and character/relation configuration for DramaQA Q2Q."""

from __future__ import annotations

import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent.parent



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

