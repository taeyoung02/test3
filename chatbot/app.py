# chatbot/app.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# 기존 파이프라인 그대로 재사용
from qdrant_openai_query import (
    PROJECT_ROOT,
    ALL_SECTIONS,
    guess_sections,
    search_raw_hits,
    aggregate_by_car,
    select_top_cars,
    make_car_context,
    classify_conversation_stage,
    answer_with_openai,
    load_conversation_history,
    save_conversation_history,
    load_cutsomer_info,
    find_json_file_for_car_id,
    load_car_json_data,
)

app = FastAPI(title="Kolon Car Sales Chatbot API")

CHATBOT_DIR = Path(PROJECT_ROOT) / "chatbot"
CAR_DATA_DIR = CHATBOT_DIR / "car_data"

# 메모리 세션 (단일 워커에서만 완전 유지됨)
SESSION: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    session_id: str = "default"
    question: str
    top_k: int = 30
    top_cars: int = 3


class ChatResponse(BaseModel):
    session_id: str
    prompt_keys: List[str]
    answer: str
    top_cars: List[Dict[str, Any]]  # [{car_id, score, chunks}, ...]


def _history_file(session_id: str) -> Path:
    if session_id == "default":
        return CHATBOT_DIR / "conversation_history.json"
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))[:80]
    return CHATBOT_DIR / f"conversation_history.{safe}.json"


def _customer_file(session_id: str) -> Path:
    if session_id == "default":
        return CHATBOT_DIR / "customer_info.json"
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))[:80]
    return CHATBOT_DIR / f"customer_info.{safe}.json"


def _make_history_summary(conversation_history: List[Dict[str, str]], n: int = 5) -> str:
    if not conversation_history:
        return ""
    recent = conversation_history[-n:]
    lines = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = (msg.get("content", "") or "")[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_persuasion_raw_sections(question: str, top_car_id: str) -> str:
    """
    persuasion 단계에서 '원문 JSON 섹션'이 필요하면 여기에 붙여서 전달.
    (recommendation에서 car_info 빈칸이던 문제도 동시에 해결)
    """
    json_filename = find_json_file_for_car_id(top_car_id, CAR_DATA_DIR)
    if not json_filename:
        return ""

    data = load_car_json_data(json_filename)
    if not data:
        return ""

    # 질문 기반 섹션만 추출 (기존 guess_sections 재사용)
    try:
        target_sections = guess_sections(question)
    except Exception:
        target_sections = ALL_SECTIONS

    selected_sections: Dict[str, Any] = {}
    for section in target_sections:
        if section == "summary":
            value = data.get("summary_text")
        else:
            value = data.get(section)
        if value is not None:
            selected_sections[section] = value

    if not selected_sections:
        return ""

    return (
        "\n\n=== RAW JSON SECTIONS (for persuasion) ===\n"
        + json.dumps(selected_sections, ensure_ascii=False, indent=2)
    )


@app.on_event("startup")
def startup():
    # 서버 시작 시 1회만 초기화(지금은 import 시점에 qdrant/openai init이 이미 됨)
    CHATBOT_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id

    # 1) 세션 상태 가져오기 (없으면 파일에서 로드)
    st = SESSION.setdefault(session_id, {})

    history_file = _history_file(session_id)
    customer_file = _customer_file(session_id)

    if "conversation_history" not in st:
        st["conversation_history"] = load_conversation_history(history_file)
    if "customer_info" not in st:
        st["customer_info"] = load_cutsomer_info(customer_file)

    conversation_history: List[Dict[str, str]] = st["conversation_history"]
    customer_info: Dict[str, Any] = st["customer_info"]

    history_summary = _make_history_summary(conversation_history)

    # 2) 단계 분류
    prompt_keys = classify_conversation_stage(
        req.question,
        history_summary,
        customer_info,
    )

    # 3) retrieval 필요 여부
    #    - recommendation: 당연히 필요
    #    - persuasion: 보통 recommendation 결과(후보차량)가 있어야 설득 가능하므로 같이 retrieval 수행
    needs_retrieval = any(k in prompt_keys for k in ("recommendation", "persuasion"))

    top_cars_payload: List[Dict[str, Any]] = []
    car_info_text = ""  # LLM에 넣을 car_info (여기서 반드시 채움)

    if needs_retrieval:
        # 3-1) 섹션 타게팅
        target_sections = guess_sections(req.question)

        # 3-2) qdrant 검색
        raw_hits = search_raw_hits(req.question, sections=target_sections, k_per_section=req.top_k)

        # 3-3) car-level ranking
        section_weights = {s: (2.0 if s in target_sections else 1.0) for s in ALL_SECTIONS}
        grouped = aggregate_by_car(raw_hits, section_weights=section_weights)
        top_cars = select_top_cars(grouped, top_n=req.top_cars)

        if top_cars:
            # 반드시 car_info 채우기 (너가 겪던 빈칸 문제 해결 포인트)
            car_info_text = "\n\n".join(make_car_context(car_id, info) for car_id, info in top_cars)

            # top car raw json 섹션도 persuasion에 유리하니 붙여줌
            top_car_id = top_cars[0][0]
            car_info_text += _build_persuasion_raw_sections(req.question, top_car_id)

            # 응답 payload용
            for car_id, info in top_cars:
                top_cars_payload.append(
                    {"car_id": car_id, "score": float(info["score"]), "chunks": len(info["chunks"])}
                )
        else:
            # 검색 결과가 없으면 빈 car_info로 진행(LLM이 fallback 답변)
            car_info_text = ""

    # 4) LLM 생성 (기존 answer_with_openai 그대로 사용)
    view_function_list = ["<rotate:target>", "<zoom-in>", "<zoom-out>"]

    answer, updated_history = answer_with_openai(
        question=req.question,
        history_summary=history_summary,
        conversation_history=conversation_history,
        customer_info=customer_info,
        prompt_keys=prompt_keys,
        car_info=car_info_text,              # ✅ 여기서 절대 빈칸으로 안 둠(검색되면)
        view_function_list=view_function_list,
    )

    # 5) 세션 상태 업데이트 + 파일 저장(서버 재시작 대비)
    st["conversation_history"] = updated_history
    st["customer_info"] = customer_info

    save_conversation_history(updated_history, history_file)
    try:
        customer_file.write_text(json.dumps(customer_info, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return ChatResponse(
        session_id=session_id,
        prompt_keys=prompt_keys,
        answer=answer,
        top_cars=top_cars_payload,
    )
