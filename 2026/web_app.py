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


def model_config_payload() -> dict[str, Any]:
    return {
        "gpu_ids": RUNTIME_GPU_IDS,
        "max_gpu_memory_gib": RUNTIME_MAX_GPU_MEMORY_GIB,
        "max_cpu_memory_gib": RUNTIME_MAX_CPU_MEMORY_GIB,
        "gpu_idle_timeout_minutes": RUNTIME_GPU_IDLE_TIMEOUT_MINUTES,
        "last_used_at": MODEL_MANAGER.last_used_at,
        "last_unloaded_at": MODEL_MANAGER.last_unloaded_at,
        "last_unload_reason": MODEL_MANAGER.last_unload_reason,
        "locked": MODEL_MANAGER.loaded,
    }


def configure_model_limits(
    gpu_ids: str | None,
    max_gpu_memory_gib: float | None,
    max_cpu_memory_gib: float | None,
    gpu_idle_timeout_minutes: float | None,
) -> None:
    global RUNTIME_GPU_IDS, RUNTIME_MAX_GPU_MEMORY_GIB
    global RUNTIME_MAX_CPU_MEMORY_GIB, RUNTIME_GPU_IDLE_TIMEOUT_MINUTES

    normalized_gpu_ids = RUNTIME_GPU_IDS
    if gpu_ids is not None:
        raw_ids = [item.strip() for item in gpu_ids.split(",") if item.strip()]
        if not raw_ids or any(not item.isdigit() for item in raw_ids):
            raise HTTPException(
                status_code=422,
                detail="GPU IDs must be comma-separated integers such as 0 or 0,1.",
            )
        if len(set(raw_ids)) != len(raw_ids):
            raise HTTPException(status_code=422, detail="GPU IDs must be unique.")
        normalized_gpu_ids = ",".join(raw_ids)

    desired_gpu_memory = (
        max_gpu_memory_gib
        if max_gpu_memory_gib is not None
        else RUNTIME_MAX_GPU_MEMORY_GIB
    )
    desired_cpu_memory = (
        max_cpu_memory_gib
        if max_cpu_memory_gib is not None
        else RUNTIME_MAX_CPU_MEMORY_GIB
    )
    desired_idle_timeout = (
        gpu_idle_timeout_minutes
        if gpu_idle_timeout_minutes is not None
        else RUNTIME_GPU_IDLE_TIMEOUT_MINUTES
    )
    placement_changed = (
        normalized_gpu_ids != RUNTIME_GPU_IDS
        or desired_gpu_memory != RUNTIME_MAX_GPU_MEMORY_GIB
        or desired_cpu_memory != RUNTIME_MAX_CPU_MEMORY_GIB
    )
    with MODEL_LOCK:
        if MODEL_MANAGER.loaded and placement_changed:
            raise HTTPException(
                status_code=409,
                detail=(
                    "The model is already loaded. Unload it before changing GPU "
                    "or memory limits."
                ),
            )
        if normalized_gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = normalized_gpu_ids
        RUNTIME_GPU_IDS = normalized_gpu_ids
        RUNTIME_MAX_GPU_MEMORY_GIB = desired_gpu_memory
        RUNTIME_MAX_CPU_MEMORY_GIB = desired_cpu_memory
        RUNTIME_GPU_IDLE_TIMEOUT_MINUTES = max(0.0, desired_idle_timeout)


def has_open_dialogue_sessions() -> bool:
    open_states = {"prepared", "generating", "awaiting_answer", "expanding", "complete"}
    with SESSIONS_LOCK:
        return any(session.status in open_states for session in SESSIONS.values())


def unload_model(reason: str, require_idle: bool = False) -> tuple[bool, str]:
    if require_idle and has_open_dialogue_sessions():
        return False, "open_dialogue"
    if not MODEL_LOCK.acquire(blocking=False):
        return False, "busy"
    try:
        if MODEL_MANAGER.agent is None:
            return True, "already_unloaded"
        if require_idle:
            timeout_seconds = RUNTIME_GPU_IDLE_TIMEOUT_MINUTES * 60.0
            if timeout_seconds <= 0:
                return False, "disabled"
            if time.time() - MODEL_MANAGER.last_used_at < timeout_seconds:
                return False, "not_idle"
        agent = MODEL_MANAGER.agent
        MODEL_MANAGER.agent = None
        agent.stop_event = None
        if hasattr(agent, "model"):
            agent.model = None
        if hasattr(agent, "processor"):
            agent.processor = None
        del agent
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (ImportError, RuntimeError, AssertionError):
            pass
        MODEL_MANAGER.last_unloaded_at = time.time()
        MODEL_MANAGER.last_unload_reason = reason
        return True, "unloaded"
    finally:
        MODEL_LOCK.release()


async def idle_unload_watcher() -> None:
    while True:
        timeout_minutes = RUNTIME_GPU_IDLE_TIMEOUT_MINUTES
        interval = 30.0 if timeout_minutes <= 0 else min(30.0, max(2.0, timeout_minutes * 15.0))
        await asyncio.sleep(interval)
        if not MODEL_MANAGER.loaded or timeout_minutes <= 0:
            continue
        if time.time() - MODEL_MANAGER.last_used_at < timeout_minutes * 60.0:
            continue
        unload_model("idle_timeout", require_idle=True)


def reset_expanded_kg_outputs() -> None:
    """Reset expanded KG files to the clean base KG when web_app starts or stops."""
    global KNOWLEDGE_BASE, KNOWLEDGE_BASE_MTIME_NS
    with KB_LOCK:
        try:
            if KG_PATH.exists():
                base_data = pipeline.load_json(KG_PATH)
                if isinstance(base_data, list):
                    pipeline.sanitize_kg_episodes(base_data)
                    pipeline.write_json_atomic(EXPANDED_KG_PATH, base_data)
                    type_map = pipeline.build_type_subtype_map(base_data)
                    pipeline.write_json_atomic(TYPE_SUBTYPE_MAP_PATH, type_map)
                    baseline_sig = pipeline.kg_graph_signature(base_data)
                    pipeline.write_kg_visualization_html(
                        EXPANDED_KG_PATH, VISUALIZATION_PATH, baseline_sig
                    )
            pipeline.write_json_atomic(EXPANSION_LOG_PATH, [])
            pipeline.save_transcript_csv(TRANSCRIPT_CSV_PATH, [], "")
        except Exception:
            pass
        KNOWLEDGE_BASE = None
        KNOWLEDGE_BASE_MTIME_NS = None


atexit.register(reset_expanded_kg_outputs)


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    reset_expanded_kg_outputs()
    get_knowledge_base()
    watcher = asyncio.create_task(idle_unload_watcher())
    try:
        yield
    finally:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
        reset_expanded_kg_outputs()



app = FastAPI(
    title="DramaQA Q2Q KG Lab",
    version="1.0.0",
    description="Interactive yes/no sub-questioning and KG expansion.",
    lifespan=app_lifespan,
)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="web/index.html is missing.")
    return FileResponse(index_path)


@app.get("/api/health")
def health() -> dict[str, Any]:
    with SESSIONS_LOCK:
        session_count = len(SESSIONS)
    return {
        "ok": True,
        "model": MODEL_NAME,
        "model_loaded": MODEL_MANAGER.loaded,
        "model_config": model_config_payload(),
        "knowledge_base_loaded": KNOWLEDGE_BASE is not None,
        "sessions": session_count,
    }


@app.post("/api/model/unload")
def unload_model_endpoint() -> dict[str, Any]:
    released, status = unload_model("manual", require_idle=False)
    if not released and status == "busy":
        raise HTTPException(
            status_code=409,
            detail=(
                "The model is generating now. Stop the active dialogue first, "
                "then unload the model."
            ),
        )
    payload = health()
    payload.update({"released": released, "unload_status": status})
    return payload


@app.get("/api/gpus")
def gpu_inventory() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "gpus": [], "error": str(exc)}
    if result.returncode != 0:
        return {
            "available": False,
            "gpus": [],
            "error": result.stderr.strip() or "nvidia-smi failed.",
        }
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            continue
        try:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_used_mib": int(fields[3]),
                }
            )
        except ValueError:
            continue
    return {"available": bool(gpus), "gpus": gpus, "error": None}


@app.get("/api/sample")
def sample() -> dict[str, str]:
    return {"question": pipeline.SAMPLE_QUESTION}


@app.post("/api/sessions")
def create_session(request: CreateSessionRequest) -> dict[str, Any]:
    question = " ".join(request.question.strip().split())
    configure_model_limits(
        request.gpu_ids,
        request.max_gpu_memory_gib,
        request.max_cpu_memory_gib,
        request.gpu_idle_timeout_minutes,
    )
    if request.min_subquestions > request.max_subquestions:
        raise HTTPException(
            status_code=422,
            detail="min_subquestions cannot exceed max_subquestions.",
        )
    explicit_ids = (
        [item.strip() for item in request.vid.split(",") if item.strip()]
        if request.vid
        else None
    )
    try:
        kb = get_knowledge_base()
        records, retrieval_method = kb.retrieve(
            question,
            top_k=request.top_scenes,
            explicit_scene_ids=explicit_ids,
        )
        if not records:
            raise RuntimeError(
                "No relevant scene was found. Enter a scene ID in Advanced options."
            )

        matched_qa = kb.find_exact_qa(question)
        shot_ranges: dict[str, tuple[int, int]] = {}
        if matched_qa and matched_qa.shot_range:
            for scene_id in matched_qa.scene_ids:
                shot_ranges[scene_id] = matched_qa.shot_range
        images = pipeline.representative_frames(
            get_image_root(),
            records,
            max_images=request.max_images,
            shot_ranges=shot_ranges,
        )
        selected_shot_ids = {
            f"{'_'.join(scene_id.split('_')[:2])}_{path.parent.name}"
            for scene_id, path in images
        }
        prior_knowledge = kb.get_prior_verified_knowledge(question, records)
        effective_min_subquestions = pipeline.compute_effective_min_subquestions(
            request.min_subquestions, prior_knowledge
        )
        prior_kg_questions = [
            str(item["sub_question"])
            for item in prior_knowledge.get("verified_subquestions", [])
            if isinstance(item, dict) and item.get("sub_question")
        ]
        context = kb.build_context(question, records, selected_shot_ids)
        messages = pipeline.initial_messages(
            question,
            context,
            images,
            effective_min_subquestions,
            request.max_subquestions,
            prior_knowledge=prior_knowledge,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    session = WebDialogueSession(
        session_id=uuid.uuid4().hex,
        question=question,
        retrieval_method=retrieval_method,
        records=records,
        images=images,
        context=context,
        messages=messages,
        min_subquestions=effective_min_subquestions,
        max_subquestions=request.max_subquestions,
        reference_answer=matched_qa.answer if matched_qa else None,
        entity_genders=dict(kb.entity_genders),
        prior_kg_questions=prior_kg_questions,
        added_relations=list(prior_knowledge.get("added_relations", [])),
        same_question_in_kg=bool(
            prior_knowledge.get("same_question_already_in_kg")
        ),
    )
    with SESSIONS_LOCK:
        while len(SESSIONS) >= MAX_SESSIONS:
            oldest_id = min(
                SESSIONS, key=lambda key: SESSIONS[key].created_at
            )
            SESSIONS.pop(oldest_id, None)
        SESSIONS[session.session_id] = session
    return session_payload(session, include_context=True)


@app.get("/api/sessions/{session_id}")
def read_session(session_id: str) -> dict[str, Any]:
    session = get_session_or_404(session_id)
    with session.lock:
        return session_payload(session, include_context=True)


@app.get("/api/sessions/{session_id}/images/{image_index}")
def session_image(session_id: str, image_index: int) -> FileResponse:
    session = get_session_or_404(session_id)
    if image_index < 0 or image_index >= len(session.images):
        raise HTTPException(status_code=404, detail="Frame not found.")
    path = session.images[image_index][1]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Frame file is missing.")
    return FileResponse(path)

