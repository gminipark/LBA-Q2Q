#!/usr/bin/env python3
"""FastAPI chat UI for the DramaQA sub-question and KG-expansion pipeline."""

from __future__ import annotations


import asyncio
import atexit
import gc
import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import run as pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
WEB_DIR = SCRIPT_DIR / "web"


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


DATASET_ROOT = env_path("LBA_DATASET_ROOT", pipeline.DEFAULT_DATASET_ROOT)
KG_PATH = env_path(
    "LBA_KG_PATH",
    DATASET_ROOT / "KG" / "AnotherMissOhQA_test_episode_scene_kg.json",
)
QA_DIR = env_path("LBA_QA_DIR", DATASET_ROOT / "QA")
OUTPUT_DIR = env_path("LBA_OUTPUT_DIR", pipeline.DEFAULT_OUTPUT_DIR)
EXPANDED_KG_PATH = OUTPUT_DIR / "expanded_kg.json"
TYPE_SUBTYPE_MAP_PATH = OUTPUT_DIR / "type_subtype_map.json"
TRANSCRIPT_CSV_PATH = OUTPUT_DIR / "input_example.csv"
EXPANSION_LOG_PATH = OUTPUT_DIR / "kg_expansion_log.json"
VISUALIZATION_PATH = OUTPUT_DIR / "visualization" / "expanded_kg.html"

MODEL_NAME = os.environ.get("LBA_MODEL", pipeline.DEFAULT_MODEL)
MODEL_DTYPE = os.environ.get("LBA_DTYPE", "auto")
MODEL_MAX_NEW_TOKENS = int(os.environ.get("LBA_MAX_NEW_TOKENS", "32768"))
MODEL_TEMPERATURE = float(os.environ.get("LBA_TEMPERATURE", "1.0"))
MODEL_TOP_P = float(os.environ.get("LBA_TOP_P", "0.95"))
MODEL_TOP_K = int(os.environ.get("LBA_TOP_K", "20"))
MODEL_ATTN = os.environ.get("LBA_ATTN_IMPLEMENTATION") or None
MODEL_CACHE_DIR_RAW = os.environ.get("LBA_MODEL_CACHE_DIR")
MODEL_CACHE_DIR = (
    Path(MODEL_CACHE_DIR_RAW).expanduser().resolve()
    if MODEL_CACHE_DIR_RAW
    else None
)
MODEL_LOCAL_ONLY = os.environ.get("LBA_LOCAL_FILES_ONLY", "0").lower() in {
    "1",
    "true",
    "yes",
}
MAX_SESSIONS = max(1, int(os.environ.get("LBA_MAX_SESSIONS", "100")))
RUNTIME_GPU_IDS = os.environ.get("CUDA_VISIBLE_DEVICES")
RUNTIME_MAX_GPU_MEMORY_GIB = (
    float(os.environ["LBA_MAX_GPU_MEMORY_GIB"])
    if os.environ.get("LBA_MAX_GPU_MEMORY_GIB")
    else None
)
RUNTIME_GPU_IDLE_TIMEOUT_MINUTES = max(0.0, float(
    os.environ.get("LBA_GPU_IDLE_TIMEOUT_MINUTES", "10"))
)
RUNTIME_MAX_CPU_MEMORY_GIB = (
    float(os.environ["LBA_MAX_CPU_MEMORY_GIB"])
    if os.environ.get("LBA_MAX_CPU_MEMORY_GIB")
    else None
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "visualization").mkdir(parents=True, exist_ok=True)


class CreateSessionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    vid: str | None = Field(default=None, max_length=1000)
    top_scenes: int = Field(default=2, ge=1, le=10)
    max_images: int = Field(default=4, ge=0, le=12)
    min_subquestions: int = Field(default=1, ge=0, le=10)
    max_subquestions: int = Field(default=5, ge=1, le=10)
    gpu_ids: str | None = Field(default=None, max_length=100)
    max_gpu_memory_gib: float | None = Field(default=None, ge=1, le=1024)
    max_cpu_memory_gib: float | None = Field(default=None, ge=1, le=4096)
    gpu_idle_timeout_minutes: float | None = Field(
        default=None, ge=0, le=1440
    )


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=16)


@dataclass
class WebDialogueSession:
    session_id: str
    question: str
    retrieval_method: str
    records: list[pipeline.SceneRecord]
    images: list[tuple[str, Path]]
    context: str
    messages: list[dict[str, Any]]
    min_subquestions: int
    max_subquestions: int
    reference_answer: str | None
    entity_genders: dict[str, str] = field(default_factory=dict)
    prior_kg_questions: list[str] = field(default_factory=list)
    added_relations: list[dict[str, str]] = field(default_factory=list)
    same_question_in_kg: bool = False
    created_at: float = field(default_factory=time.time)
    status: str = "prepared"
    current_sub_question: str | None = None
    transcript: list[dict[str, str]] = field(default_factory=list)
    final_decision: pipeline.Decision | None = None
    expansion_results: list[dict[str, Any]] = field(default_factory=list)
    expansion_error: str | None = None
    cancel_requested: bool = False
    stop_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

