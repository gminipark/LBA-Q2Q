"""Interactive CLI binary dialogue runner and sample preview helper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from q2q.agent import QwenSubQuestionAgent
from q2q.config import SAMPLE_QUESTION
from q2q.frames import representative_frames
from q2q.kb import DramaQAKnowledgeBase
from q2q.prompts import build_turn_followup_prompt, text_content
from q2q.schemas import Decision, normalize_binary_answer


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

