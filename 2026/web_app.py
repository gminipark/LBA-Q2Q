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


class ModelManager:
    """Lazily loads one model and serializes use of its mutable session fields."""

    def __init__(self) -> None:
        self.agent: pipeline.QwenSubQuestionAgent | None = None
        self.last_used_at = time.time()
        self.last_unloaded_at: float | None = None
        self.last_unload_reason: str | None = None

    def mark_used(self) -> None:
        self.last_used_at = time.time()


    @property
    def loaded(self) -> bool:
        return self.agent is not None

    def get(
        self,
        question: str,
        reference_answer: str | None,
        entity_genders: dict[str, str] | None = None,
        prior_kg_questions: list[str] | None = None,
        added_relations: list[dict[str, str]] | None = None,
        same_question_in_kg: bool = False,
    ) -> pipeline.QwenSubQuestionAgent:
        if self.agent is None:
            self.agent = pipeline.QwenSubQuestionAgent(
                model_name=MODEL_NAME,
                dtype=MODEL_DTYPE,
                max_new_tokens=MODEL_MAX_NEW_TOKENS,
                temperature=MODEL_TEMPERATURE,
                top_p=MODEL_TOP_P,
                top_k=MODEL_TOP_K,
                attn_implementation=MODEL_ATTN,
                cache_dir=MODEL_CACHE_DIR,
                local_files_only=MODEL_LOCAL_ONLY,
                original_question=question,
                reference_answer=reference_answer,
                max_gpu_memory_gib=RUNTIME_MAX_GPU_MEMORY_GIB,
                max_cpu_memory_gib=RUNTIME_MAX_CPU_MEMORY_GIB,
                entity_genders=entity_genders,
                prior_kg_questions=prior_kg_questions or (),
                added_relations=added_relations or (),
                same_question_in_kg=same_question_in_kg,
            )
        self.agent.original_question = question
        self.agent.reference_answer = reference_answer
        self.agent.entity_genders = dict(entity_genders or {})
        self.agent.prior_kg_questions = list(prior_kg_questions or ())
        self.agent.added_relations = list(added_relations or ())
        self.agent.same_question_in_kg = bool(same_question_in_kg)
        self.mark_used()
        return self.agent


SESSIONS: dict[str, WebDialogueSession] = {}
SESSIONS_LOCK = threading.RLock()
MODEL_LOCK = threading.RLock()
KB_LOCK = threading.RLock()
MODEL_MANAGER = ModelManager()
KNOWLEDGE_BASE: pipeline.DramaQAKnowledgeBase | None = None
KNOWLEDGE_BASE_MTIME_NS: int | None = None
IMAGE_ROOT: Path | None = None


def _expanded_kg_mtime_ns() -> int | None:
    try:
        if EXPANDED_KG_PATH.exists():
            return EXPANDED_KG_PATH.stat().st_mtime_ns
    except OSError:
        pass
    return None


def get_knowledge_base() -> pipeline.DramaQAKnowledgeBase:
    global KNOWLEDGE_BASE, KNOWLEDGE_BASE_MTIME_NS
    with KB_LOCK:
        current_mtime = _expanded_kg_mtime_ns()
        if KNOWLEDGE_BASE is None or current_mtime != KNOWLEDGE_BASE_MTIME_NS:
            KNOWLEDGE_BASE = pipeline.DramaQAKnowledgeBase(
                kg_path=KG_PATH,
                qa_dir=QA_DIR,
                expanded_kg_path=EXPANDED_KG_PATH,
                expansion_log_path=EXPANSION_LOG_PATH,
            )
            KNOWLEDGE_BASE_MTIME_NS = _expanded_kg_mtime_ns()
        return KNOWLEDGE_BASE


def get_image_root() -> Path:
    global IMAGE_ROOT
    with KB_LOCK:
        if IMAGE_ROOT is None:
            requested = os.environ.get("LBA_IMAGE_ROOT")
            IMAGE_ROOT = pipeline.resolve_image_root(
                Path(requested) if requested else None
            )
        return IMAGE_ROOT


def get_session_or_404(session_id: str) -> WebDialogueSession:
    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dialogue session not found.")
    return session


def public_outputs(session: WebDialogueSession) -> dict[str, str] | None:
    if session.status != "expanded":
        return None
    version = (
        str(int(VISUALIZATION_PATH.stat().st_mtime_ns))
        if VISUALIZATION_PATH.exists()
        else str(int(time.time() * 1000))
    )
    return {
        "expanded_kg": f"/output/expanded_kg.json?v={version}",
        "type_subtype_map": f"/output/type_subtype_map.json?v={version}",
        "transcript_csv": f"/output/input_example.csv?v={version}",
        "expansion_log": f"/output/kg_expansion_log.json?v={version}",
        "visualization": (
            f"/output/visualization/expanded_kg.html?v={version}"
        ),
    }


def session_payload(
    session: WebDialogueSession, *, include_context: bool = False
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": session.session_id,
        "question": session.question,
        "status": session.status,
        "retrieval_method": session.retrieval_method,
        "scene_ids": [record.scene_id for record in session.records],
        "images": [
            {
                "scene_id": scene_id,
                "url": f"/api/sessions/{session.session_id}/images/{index}",
            }
            for index, (scene_id, _) in enumerate(session.images)
        ],
        "current_sub_question": session.current_sub_question,
        "dialogue": list(session.transcript),
        "turn": len(session.transcript) + (
            1 if session.current_sub_question else 0
        ),
        "min_subquestions": session.min_subquestions,
        "max_subquestions": session.max_subquestions,
        "final": (
            asdict(session.final_decision) if session.final_decision else None
        ),
        "kg_expansion": list(session.expansion_results),
        "kg_expansion_error": session.expansion_error,
        "cancel_requested": session.cancel_requested,
        "kg_expansion_outputs": public_outputs(session),
        "model": MODEL_NAME,
        "model_loaded": MODEL_MANAGER.loaded,
        "model_config": model_config_payload(),
    }
    if include_context:
        try:
            payload["kg_context"] = json.loads(session.context)
        except json.JSONDecodeError:
            payload["kg_context"] = session.context
    return payload


def model_for(session: WebDialogueSession) -> pipeline.QwenSubQuestionAgent:
    return MODEL_MANAGER.get(
        session.question,
        session.reference_answer,
        entity_genders=session.entity_genders,
        prior_kg_questions=session.prior_kg_questions,
        added_relations=session.added_relations,
        same_question_in_kg=session.same_question_in_kg,
    )


class SessionAgentContext:
    def __init__(self, session: WebDialogueSession) -> None:
        self.session = session
        self.agent: pipeline.QwenSubQuestionAgent | None = None

    def __enter__(self) -> pipeline.QwenSubQuestionAgent:
        MODEL_LOCK.acquire()
        try:
            self.agent = model_for(self.session)
            self.agent.stop_event = self.session.stop_event
            if self.session.stop_event.is_set():
                raise pipeline.GenerationCancelled(
                    "Generation was cancelled by the user."
                )
            return self.agent
        except BaseException:
            if self.agent is not None:
                self.agent.stop_event = None
            MODEL_MANAGER.mark_used()
            MODEL_LOCK.release()
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.agent is not None:
            self.agent.stop_event = None
            MODEL_MANAGER.mark_used()
        MODEL_LOCK.release()

