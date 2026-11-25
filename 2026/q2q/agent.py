"""Multimodal Qwen VLM agent wrapper for binary sub-question generation."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from threading import Event
from typing import Any, Sequence

from q2q.prompts import text_content
from q2q.schemas import Decision, GenerationCancelled
from q2q.validation import (
    json_objects_from_text,
    normalize_model_sub_question,
    parse_decision,
    safe_fallback_sub_question,
    validate_decision,
)


class QwenSubQuestionAgent:
    def __init__(
        self,
        model_name: str,
        dtype: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        attn_implementation: str | None,
        cache_dir: Path | None,
        local_files_only: bool,
        original_question: str,
        reference_answer: str | None,
        max_gpu_memory_gib: float | None = None,
        max_cpu_memory_gib: float | None = None,
        entity_genders: dict[str, str] | None = None,
        prior_kg_questions: Sequence[str] = (),
        added_relations: Sequence[dict[str, str]] = (),
        same_question_in_kg: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen multimodal inference requires torch, transformers, "
                "accelerate, torchvision, and pillow. "
                f"(python={sys.executable}, error={exc})"
            ) from exc

        model_name_lower = model_name.lower()
        is_qwen35 = any(
            marker in model_name_lower
            for marker in (
                "qwen3.5",
                "qwen3_5",
                "qwen3.6",
                "qwen3_6",
                "qwen3.8",
                "qwen3_8",
            )
        )
        try:
            if is_qwen35:
                from transformers import Qwen3_5ForConditionalGeneration

                model_class = Qwen3_5ForConditionalGeneration
            else:
                from transformers import Qwen3VLForConditionalGeneration

                model_class = Qwen3VLForConditionalGeneration
        except ImportError as exc:
            requirement = (
                "the latest Transformers main branch"
                if is_qwen35
                else "transformers>=4.57.0"
            )
            raise RuntimeError(
                f"{model_name} requires {requirement}. Activate the LBA Conda "
                "environment before running this script. "
                f"(python={sys.executable}, error={exc})"
            ) from exc

        dtype_map = {
            "auto": "auto",
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        model_kwargs: dict[str, Any] = {
            "dtype": dtype_map[dtype],
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            model_kwargs["cache_dir"] = str(cache_dir)
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if max_gpu_memory_gib is not None or max_cpu_memory_gib is not None:
            max_memory: dict[int | str, str] = {}
            if max_gpu_memory_gib is not None:
                if max_gpu_memory_gib <= 0:
                    raise ValueError("GPU memory limit must be greater than zero.")
                gpu_count = torch.cuda.device_count()
                if gpu_count == 0:
                    raise RuntimeError(
                        "A GPU memory limit was requested, but no visible CUDA GPU "
                        "was found. Check CUDA_VISIBLE_DEVICES."
                    )
                gpu_limit = f"{max_gpu_memory_gib:g}GiB"
                max_memory.update(
                    {device_index: gpu_limit for device_index in range(gpu_count)}
                )
            if max_cpu_memory_gib is not None:
                if max_cpu_memory_gib <= 0:
                    raise ValueError("CPU memory limit must be greater than zero.")
                max_memory["cpu"] = f"{max_cpu_memory_gib:g}GiB"
            model_kwargs["max_memory"] = max_memory

        print(f"[model] Loading {model_name} ...", flush=True)
        self.model = model_class.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        processor_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            processor_kwargs["cache_dir"] = str(cache_dir)
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.torch = torch
        self.is_qwen35 = is_qwen35
        enable_thinking = os.environ.get("LBA_ENABLE_THINKING", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self.requires_complete_thinking = (
            enable_thinking and (is_qwen35 or "thinking" in model_name_lower)
        )
        self.chat_template_kwargs = (
            {"enable_thinking": enable_thinking} if is_qwen35 else {}
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.original_question = original_question
        self.reference_answer = reference_answer
        self.entity_genders: dict[str, str] = dict(entity_genders or {})
        self.prior_kg_questions: list[str] = list(prior_kg_questions)
        self.added_relations: list[dict[str, str]] = list(added_relations)
        self.same_question_in_kg: bool = bool(same_question_in_kg)
        self.stop_event: Event | None = None

    def _raise_if_cancelled(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise GenerationCancelled("Generation was cancelled by the user.")

    def _run_model_generate(
        self, inputs: Any, **generation_kwargs: Any
    ) -> Any:
        self._raise_if_cancelled()
        if self.stop_event is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList

            stop_event = self.stop_event

            class EventStoppingCriteria(StoppingCriteria):
                def __call__(
                    self, input_ids: Any, scores: Any, **kwargs: Any
                ) -> bool:
                    return stop_event.is_set()

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [EventStoppingCriteria()]
            )
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        self._raise_if_cancelled()
        return generated

    def generate(self, messages: Sequence[dict[str, Any]]) -> str:
        inputs = self.processor.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self.chat_template_kwargs,
        )
        inputs = inputs.to(self.model.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        generated = self._run_model_generate(inputs, **generation_kwargs)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated)
        ]
        raw_with_special = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if "</think>" in raw_with_special and "</think>" not in decoded:
            think_part, visible_part = raw_with_special.rsplit("</think>", 1)
            special_tokens = getattr(
                getattr(self.processor, "tokenizer", None),
                "all_special_tokens",
                [],
            )
            for tok in special_tokens:
                if tok != "</think>":
                    think_part = think_part.replace(tok, "")
                    visible_part = visible_part.replace(tok, "")
            return f"{think_part.strip()}\n</think>\n\n{visible_part.strip()}"
        return decoded

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        enable_thinking: bool = False,
    ) -> str:
        template_kwargs = (
            {"enable_thinking": enable_thinking} if self.is_qwen35 else {}
        )
        messages = [
            {"role": "user", "content": text_content(prompt)}
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        inputs = inputs.to(self.model.device)
        generated = self._run_model_generate(
            inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


    def complete_structured_output(
        self,
        messages: Sequence[dict[str, Any]],
        reasoning: str,
        force_final: bool,
        can_finish: bool = False,
    ) -> str:
        print(
            "[model] Completing structured JSON output ...",
            file=sys.stderr,
            flush=True,
        )
        reasoning = reasoning.split("</think>", 1)[0].strip()
        if force_final:
            prefix = "{\"done\": true, \"final_answer\": \""
        elif can_finish:
            prefix = "{\"done\": "
        else:
            prefix = "{\"done\": false, \"sub_question\": \""
        prefilled_messages = list(messages) + [
            {
                "role": "assistant",
                "reasoning_content": reasoning.strip(),
                "content": text_content(prefix),
            }
        ]
        inputs = self.processor.apply_chat_template(
            prefilled_messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
            **self.chat_template_kwargs,
        )
        inputs = inputs.to(self.model.device)
        visible_token_limit = min(128, max(64, self.max_new_tokens))
        generated = self._run_model_generate(
            inputs, max_new_tokens=visible_token_limit, do_sample=False
        )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated)
        ]
        continuation = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        combined = prefix + continuation
        objects = json_objects_from_text(combined)
        if not objects:
            objects = json_objects_from_text(continuation)
        chose_final = force_final
        evidence_list: list[str] = []
        value: str | None = None
        if objects:
            candidate = objects[-1]
            done_val = candidate.get("done")
            if isinstance(done_val, str):
                done_val = done_val.strip().lower() == "true"
            if not force_final and can_finish and isinstance(done_val, bool):
                chose_final = done_val
            key = "final_answer" if chose_final else "sub_question"
            alt_key = "answer" if chose_final else "question"
            raw_val = candidate.get(key) or candidate.get(alt_key)
            if isinstance(raw_val, str) and raw_val.strip():
                value = raw_val.strip()
            if chose_final and isinstance(candidate.get("evidence"), list):
                evidence_list = [
                    str(item).strip()
                    for item in candidate["evidence"]
                    if str(item).strip()
                ]
        if value is None:
            if not force_final and can_finish:
                chose_final = continuation.lstrip().lower().startswith("true")
            fragment = continuation.lstrip()
            target_key = "\"final_answer\"" if chose_final else "\"sub_question\""
            if target_key in fragment:
                fragment = fragment.split(target_key, 1)[1].lstrip()
                if fragment.startswith(":"):
                    fragment = fragment[1:].lstrip()
            if fragment.startswith("\""):
                fragment = fragment[1:]
            escaped = False
            end_index: int | None = None
            for index, character in enumerate(fragment):
                if character == "\"" and not escaped:
                    end_index = index
                    break
                if character == "\\" and not escaped:
                    escaped = True
                else:
                    escaped = False
            raw_value = fragment[:end_index] if end_index is not None else fragment
            raw_value = raw_value.strip().splitlines()[0] if raw_value.strip() else ""
            try:
                value = json.loads("\"" + raw_value + "\"") if raw_value else ""
            except json.JSONDecodeError:
                value = raw_value.replace("\\\"", "\"").strip()
        if not value:
            value = (
                "Unable to determine from the available evidence."
                if chose_final
                else "Is the relevant evidence visible in the selected frames?"
            )
        if not chose_final:
            normalized_value = normalize_model_sub_question(
                value, self.entity_genders
            )
            if normalized_value is None:
                print(
                    "[model] Discarded malformed or non-English sub-question output.",
                    file=sys.stderr, flush=True
                )
                value = safe_fallback_sub_question(())
            else:
                value = normalized_value
        payload: dict[str, Any] = (
            {"done": True, "final_answer": value}
            if chose_final
            else {"done": False, "sub_question": value}
        )
        if chose_final and evidence_list:
            payload["evidence"] = evidence_list
        return "</think>\n\n" + json.dumps(payload, ensure_ascii=False)


    def decide(
        self,
        messages: Sequence[dict[str, Any]],
        asked_questions: Sequence[str],
        min_subquestions: int,
        force_final: bool = False,
        answers: Sequence[str] = (),
    ) -> Decision:
        can_finish = (
            len(asked_questions) >= min_subquestions
            and ("Yes" in answers or not answers)
        )
        prefer_final_from_kg = (
            force_final
            or (min_subquestions <= 0 and self.same_question_in_kg and not asked_questions)
        )
        raw = self.generate(messages)
        if self.requires_complete_thinking and "</think>" not in raw:
            raw = self.complete_structured_output(
                messages, raw, prefer_final_from_kg, can_finish=can_finish
            )
        error: str | None
        try:
            decision = parse_decision(raw)
            if decision.sub_question:
                decision.sub_question = (
                    normalize_model_sub_question(
                        decision.sub_question, self.entity_genders
                    )
                    or decision.sub_question
                )
            error = validate_decision(
                decision,
                asked_questions,
                min_subquestions,
                force_final,
                self.original_question,
                self.reference_answer,
                entity_genders=self.entity_genders,
                prior_kg_questions=self.prior_kg_questions,
                added_relations=self.added_relations,
                answers=answers,
            )
        except ValueError as exc:
            decision = Decision(done=False)
            error = str(exc)
        if error is None:
            return decision

        repair_force_final = force_final or (
            can_finish and self.same_question_in_kg
        )
        repair_instruction = (
            f"Your last response was invalid: {error} "
            "Correct it now. Check the KG character genders and existing relations carefully. "
            "Any sub-question must use ASCII English, begin with an English binary auxiliary, "
            "respect character genders in the KG, never repeat a question/relation already in "
            "the KG, and contain at most 25 words. Output only one valid JSON object with no explanation."
        )
        repair_messages = list(messages) + [
            {"role": "assistant", "content": text_content(raw)},
            {"role": "user", "content": text_content(repair_instruction)},
        ]
        repaired_raw = self.generate(repair_messages)
        if self.requires_complete_thinking and "</think>" not in repaired_raw:
            repaired_raw = self.complete_structured_output(
                repair_messages,
                repaired_raw,
                repair_force_final,
                can_finish=can_finish,
            )
        try:
            repaired = parse_decision(repaired_raw)
        except ValueError:
            repaired_raw = self.complete_structured_output(
                repair_messages,
                repaired_raw,
                repair_force_final,
                can_finish=can_finish,
            )
            repaired = parse_decision(repaired_raw)
        if repaired.sub_question:
            repaired.sub_question = (
                normalize_model_sub_question(
                    repaired.sub_question, self.entity_genders
                )
                or repaired.sub_question
            )
        repaired_error = validate_decision(
            repaired,
            asked_questions,
            min_subquestions,
            force_final,
            self.original_question,
            self.reference_answer,
            entity_genders=self.entity_genders,
            prior_kg_questions=self.prior_kg_questions,
            added_relations=self.added_relations,
            answers=answers,
        )
        if repaired_error and not force_final:
            if can_finish and self.same_question_in_kg:
                final_raw = self.complete_structured_output(
                    repair_messages, repaired_raw, True, can_finish=True
                )
                final_decision = parse_decision(final_raw)
                if final_decision.done and final_decision.final_answer:
                    if not final_decision.evidence and self.added_relations:
                        final_decision.evidence = [
                            f"Verified in expanded KG: {rel.get('fact', '')}"
                            for rel in self.added_relations[:3]
                            if rel.get("fact")
                        ]
                    return final_decision

            normalized_question = normalize_model_sub_question(
                repaired.sub_question or "", self.entity_genders
            )
            if normalized_question:
                normalized_decision = Decision(
                    done=False, sub_question=normalized_question
                )
                normalized_error = validate_decision(
                    normalized_decision,
                    asked_questions,
                    min_subquestions,
                    force_final,
                    self.original_question,
                    self.reference_answer,
                    entity_genders=self.entity_genders,
                    prior_kg_questions=self.prior_kg_questions,
                    added_relations=self.added_relations,
                    answers=answers,
                )
                if normalized_error is None:
                    print(
                        "[model] Normalized the sub-question to yes/no form.",
                        file=sys.stderr, flush=True
                    )
                    return normalized_decision

            fallback_decision = Decision(
                done=False,
                sub_question=safe_fallback_sub_question(
                    list(asked_questions) + list(self.prior_kg_questions)
                ),
            )
            fallback_error = validate_decision(
                fallback_decision,
                asked_questions,
                min_subquestions,
                force_final,
                self.original_question,
                self.reference_answer,
                entity_genders=self.entity_genders,
                prior_kg_questions=self.prior_kg_questions,
                added_relations=self.added_relations,
                answers=answers,
            )
            if fallback_error is None:
                print(
                    "[model] Replaced an invalid sub-question with a safe yes/no "
                    "evidence question.",
                    file=sys.stderr, flush=True
                )
                return fallback_decision
        if repaired_error:
            raise RuntimeError(
                "The model failed the yes/no output constraint after one repair: "
                f"{repaired_error}"
            )
        return repaired

