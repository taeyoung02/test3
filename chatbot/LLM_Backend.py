from __future__ import annotations
import argparse
from calendar import c
import json
import os
import textwrap
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
from collections import defaultdict
import asyncio

from openai import AsyncOpenAI
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, Filter, FieldCondition, MatchValue

from prompts import (
    PROMPT_REGISTRY,
    PRE_STAGE_CLASSIFICATION_SYSTEM,
    PRE_STAGE_CLASSIFICATION_USER_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, "chatbot", ".env"))

OPENAI_MODEL_EMBED = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
OPENAI_MODEL_CHAT = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "used_car_info")

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant_kwargs = {}
if url := os.getenv("QDRANT_URL"):
    qdrant_kwargs["url"] = url
else:
    qdrant_kwargs["host"] = os.getenv("QDRANT_HOST", "localhost")
    qdrant_kwargs["port"] = int(os.getenv("QDRANT_PORT", "6333"))

if api_key := os.getenv("QDRANT_API_KEY"):
    qdrant_kwargs["api_key"] = api_key

qdrant_client = QdrantClient(**qdrant_kwargs)


# ---------------------------------------------------------------------------
# Car ID to JSON file mapping
# ---------------------------------------------------------------------------

def find_json_file_for_car_id(car_id: str, car_data_dir: Path) -> Optional[str]:
    """
    car_id에 해당하는 JSON 원문 파일명을 찾습니다.
    
    Args:
        car_id: 차량 ID (예: "E300_2019_001")
        car_data_dir: car_data 디렉토리 경로
        
    Returns:
        파일명 (예: "car_info1.json") 또는 None
    """
    # vector 파일 제외하고 원문 JSON 파일만 검색
    json_files = sorted(car_data_dir.glob("car_info*.json"))
    json_files = [f for f in json_files if "_vector" not in f.name]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 단일 객체 형태 (car_info.json 같은 경우)
            if isinstance(data, dict):
                vehicle_id = data.get("vehicle_id")
                if vehicle_id == car_id:
                    return json_file.name
            
            # 배열 형태 (car_info1.json 같은 경우)
            elif isinstance(data, list) and len(data) > 0:
                # 배열 내 모든 항목을 확인하여 car_id 매칭
                for item in data:
                    if isinstance(item, dict):
                        payload = item.get("payload", {})
                        file_car_id = payload.get("car_id")
                        if file_car_id == car_id:
                            return json_file.name
        except Exception as e:
            # 파일 읽기 실패 시 다음 파일로
            continue
    
    return None


def get_car_json_files(top_cars: List[tuple]) -> Dict[str, Optional[str]]:
    """
    추천된 차량들의 car_id에 해당하는 JSON 파일명을 찾아 반환합니다.
    
    Args:
        top_cars: [(car_id, info), ...] 형태의 리스트
        
    Returns:
        {car_id: json_filename} 딕셔너리
    """
    car_data_dir = Path(PROJECT_ROOT) / "chatbot" / "car_data"
    car_json_files = {}
    
    for car_id, _ in top_cars:
        json_filename = find_json_file_for_car_id(car_id, car_data_dir)
        car_json_files[car_id] = json_filename
        if json_filename:
            print(f"  → {car_id}: {json_filename}")
        else:
            print(f"  → {car_id}: 파일을 찾을 수 없습니다")
    
    return car_json_files


def load_car_json_data(json_filename: str) -> Optional[Dict[str, Any]]:
    """
    JSON 파일을 로드하여 차량 데이터를 반환합니다.
    
    Args:
        json_filename: JSON 파일명 (예: "car_info1.json")
        
    Returns:
        차량 데이터 딕셔너리 또는 None
    """
    if not json_filename:
        return None
    
    car_data_dir = Path(PROJECT_ROOT) / "chatbot" / "car_data"
    json_file_path = car_data_dir / json_filename
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"JSON 파일 로드 실패: {json_filename}, 오류: {e}")
        return None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_text(text: str) -> List[float]:
    resp = openai_client.embeddings.create(
        model=OPENAI_MODEL_EMBED,
        input=text,
    )
    return resp.data[0].embedding


# ---------------------------------------------------------------------------
# Section targeting
# ---------------------------------------------------------------------------

ALL_SECTIONS = [
    "summary",
    "basic_info",
    "special_usage_history",
    "special_accident_history",
    "registration_change_history",
    "insurance_accident_history",
    "options",
    "seller_info",
    "inspection_record",
]

# ---------------------------------------------------------------------------
# Vector retrieval (chunk-level)
# ---------------------------------------------------------------------------

def search_raw_hits(question: str, sections: List[str], k_per_section: int = 30) -> List[ScoredPoint]:
    vector = embed_text(question)
    hits: List[ScoredPoint] = []
    
    if sections:
        for section in sections:
            section_filter = Filter(
                must=[
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section),
                    )
                ]
            )

            res = qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vector,
                limit=k_per_section,
                query_filter=section_filter,
            )
            hits.extend(res.points)
    else:
        res = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=k_per_section
        )
        hits.extend(res.points)
    
    return hits

# ---------------------------------------------------------------------------
# Car-level ranking
# ---------------------------------------------------------------------------

def aggregate_by_car(hits: Iterable[ScoredPoint]) -> Dict[str, Dict[str, Any]]:
    """
    Return:
        {
            car_id: {
                "score": aggregated score,
                "chunks": [ScoredPoint, ...]
            },
            ...
        }
    """
    grouped = defaultdict(lambda: {"score": 0.0, "chunks": []})
    
    for h in hits:
        payload = h.payload or {}
        car_id = payload.get("car_id")
        if not car_id:
            continue

        grouped[car_id]["score"] += float(h.score)
        grouped[car_id]["chunks"].append(h)

    return grouped


def select_top_cars(grouped, top_n=3):
    """Sort cars by total score."""
    cars = sorted(grouped.items(), key=lambda x: x[1]["score"], reverse=True)
    return cars[:top_n]


# ---------------------------------------------------------------------------
# 해야할 사전 작업 추론
# ---------------------------------------------------------------------------
def classify_pre_task(
    question: str,
    history_summary: List[Dict[str, str]] = None,
    customer_info: Dict[str, Any] = None
) -> List[str]:
    """
    대화 단계를 분류하여 필요한 프롬프트 키 리스트를 반환합니다.
    
    Args:
        question: 사용자 질문
        history_summary: 대화 히스토리 요약
        customer_info: 고객 정보 (예산, 선호도 등)
        
    Returns:
        프롬프트 키 리스트 (예: ["guess_sections"] 또는 ["user_info_update", "vector_db_query"])
    """
    if history_summary is None:
        history_summary = []
    if customer_info is None:
        customer_info = {}
    

    user_prompt = PRE_STAGE_CLASSIFICATION_USER_TEMPLATE.format(
        question=question,
        history_summary=history_summary if history_summary else "(대화 없음)",
        customer_info=customer_info
    )
    
    try:
        resp = openai_client.responses.create(
            model=OPENAI_MODEL_CHAT,
            input=[
                {"role": "system", "content": PRE_STAGE_CLASSIFICATION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.output_text.strip()
        parts = [p.strip() for p in text.replace(" ", "").split(",") if p.strip()]
        
        # 유효성 검증
        valid_keys = [k for k in parts if k in PROMPT_REGISTRY]
        
        if not valid_keys:
            # 기본값: 사전작업이 필요하지 않은 경우
            return None
        
        print(f"분류된 프롬프트: {valid_keys}")
        return valid_keys
        
    except Exception as e:
        print(f"단계 분류 실패: {e}")
        return None


# ---------------------------------------------------------------------------
# Prompt Combination
# ---------------------------------------------------------------------------
def build_combined_prompt(
    prompt_keys: List[str],
    question: str,
    context: Dict[str, Any] = None
) -> tuple[str, str]:
    """
    여러 프롬프트를 조합하여 최종 프롬프트를 생성합니다.
    
    Args:
        prompt_keys: 사용할 프롬프트 키 리스트 (예: ["recommendation", "persuasion"])
        question: 사용자 질문
        context: 컨텍스트 정보 (cars, history, customer_info 등)
        
    Returns:
        (system_prompt, user_prompt) 튜플
    """
    if context is None:
        context = {}
    
    # System 프롬프트 조합 (여러 프롬프트의 system을 결합)
    system_parts = []
    for key in prompt_keys:
        if key in PROMPT_REGISTRY:
            system_parts.append(PROMPT_REGISTRY[key]["system"])
    
    if not system_parts:
        # 기본 프롬프트
        system_parts.append(PROMPT_REGISTRY["recommendation"]["system"])
    
    combined_system = "\n\n---\n\n".join(system_parts)
    
    # User 프롬프트 생성 (첫 번째 프롬프트의 템플릿 사용)
    primary_key = prompt_keys[0] if prompt_keys else "recommendation"
    user_template = PROMPT_REGISTRY[primary_key]["user_template"]
    
    # 컨텍스트 변수 준비
    history_summary =  f"\n\n=== 이전 대화 내용 ===\n" + context["history_summary"]
    
    customer_info = ""
    if context.get("customer_info"):
        customer_info = json.dumps(
            context["customer_info"], ensure_ascii=False, indent=2
        )
    
    car_info = context.get("car_info", "")


    view_function_list = context.get("view_function_list", "")
    
    # 템플릿 변수 채우기
    user_prompt = user_template.format(
        question=question,
        history_summary=history_summary,
        customer_info=customer_info,
        car_info=car_info,
        view_function_list=view_function_list,
    )
    
    return combined_system, user_prompt


# ---------------------------------------------------------------------------
# Build LLM context
# ---------------------------------------------------------------------------

def make_car_context(car_id: str, car_info: Dict[str, Any]) -> str:
    chunks = []
    for h in car_info["chunks"]:
        payload = h.payload or {}
        
        # 안전하게 문자열로 변환 (set 등 JSON 직렬화 불가능한 타입 처리)
        def safe_str(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, (str, int, float, bool)):
                return str(value)
            if isinstance(value, (list, tuple, set)):
                return ", ".join(safe_str(item) for item in value)
            if isinstance(value, dict):
                return str(value)  # dict는 str()로 변환
            return str(value)
        
        section = safe_str(payload.get("section", "unknown"))
        nl = payload.get("nl")
        
        if not nl:
            raw = payload.get("raw")
            if raw:
                nl = textwrap.shorten(safe_str(raw), width=500)
        
        nl_str = safe_str(nl) if nl else "(no description)"

        chunks.append(
            f"[section: {section}] score={h.score:.4f}\n{nl_str}"
        )

    return textwrap.dedent(
        f"""
        === Car ID: {car_id} ===
        Total relevance score={car_info["score"]:.4f}

        {chr(10).join(chunks)}
        """
    ).strip()

# ---------------------------------------------------------------------------
# 대화 히스토리 관리
# ---------------------------------------------------------------------------
def load_conversation_history(path: Path) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []  # 빈 파일이면 대화 없음으로 취급
    try:
        data = json.loads(txt)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        # 깨진 파일이면 백업하고 초기화(선택)
        p.rename(p.with_suffix(p.suffix + ".broken"))
        return []

def load_cutsomer_info(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return {}
    return json.loads(txt)



def save_conversation_history(history: List[Dict[str, str]], history_file: Path) -> None:
    """대화 히스토리를 파일에 저장합니다."""
    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"히스토리 저장 실패: {e}")






# elif "guess_sections" in prompt_keys:
#     try:
#         # 질문 기반으로 필요한 섹션 먼저 분류 (기존 섹션 분류 프롬프트 재사용)
#         target_sections = pre_result["target_sections"]
#         print(target_sections)
#         # car_json_data 구조에서 섹션별 raw 값만 추출
#         selected_sections: Dict[str, Any] = {}
#         for section in target_sections:
#             # summary 섹션은 summary_text 필드 매핑
#             if section == "summary":
#                 value = car_json_data.get("summary_text")
#             else:
#                 value = car_json_data.get(section)
#             print(value)
#             if value is not None:
#                 selected_sections[section] = value
#         print(selected_sections)
#         # 없으면 전체를 넣지는 않고 빈 객체로 둔다
#         car_info = json.dumps(
#             selected_sections, ensure_ascii=False, indent=2
#         )
#     except Exception as e:
#         print(f"설득 단계 car_json_data 처리 중 오류: {e}")


def guess_sections(question: str, history_summary: List[Dict[str, str]] = None) -> List[str]:
    user_prompt = PROMPT_REGISTRY["guess_sections"]["user_template"].format(question=question, history_summary=history_summary) 

    resp = openai_client.responses.create(
        model=OPENAI_MODEL_CHAT,
        input=[
            {"role": "system", "content": PROMPT_REGISTRY["guess_sections"]["system"]},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = resp.output_text.strip()
    parts = [p.strip() for p in text.replace(" ", "").split(",") if p.strip()]
    valid = [p for p in parts if p in ALL_SECTIONS]
    print(f"sections: {valid}")
    return valid or ALL_SECTIONS

async def update_cutsomer_info(question: str, customer_info: Dict[str, Any]) -> None:
    messages = []
    # 시스템 메시지는 항상 첫 번째에 (히스토리에는 포함하지 않음)
    messages.append({"role": "system", "content": PROMPT_REGISTRY['user_info_update']['system']})
    messages.append({"role": "user", "content": PROMPT_REGISTRY['user_info_update']['user_template'].format(customer_info=customer_info, question=question)})

    resp = openai_client.responses.create(
        model=OPENAI_MODEL_CHAT,
        input=messages
    )
    answer = resp.output_text
    
    customer_info_file = Path(PROJECT_ROOT) / "chatbot" / "customer_info.json"
    try:
        parsed_obj = json.loads(answer)
    except json.JSONDecodeError as e:
        print(f"유저 정보 파싱 실패: {e}")
        return
    try:
        with open(customer_info_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_obj, f, ensure_ascii=False)
    except Exception as e:
        print(f"유저 정보 저장 실패: {e}")

async def generate_query_sentence(question: str, customer_info: Dict[str, Any], history_summary: List[Dict[str, str]] = None):
    messages = []
    # 시스템 메시지는 항상 첫 번째에 (히스토리에는 포함하지 않음)
    messages.append({"role": "system", "content": PROMPT_REGISTRY['vector_db_query']['system']})
    messages.append({"role": "user", "content": PROMPT_REGISTRY['vector_db_query']['user_template'].format(customer_info=customer_info, question=question, history_summary=history_summary)})

    resp = openai_client.responses.create(
        model=OPENAI_MODEL_CHAT,
        input=messages
    )
    answer = resp.output_text
    return answer
    


async def run_pre_prompt(prompt_keys: List[str], question:str, customer_info: Dict[str, Any],
                        history_summary: List[Dict[str, str]] = None) -> Dict[str,str]:
    target_sections = []
    if 'guess_sections' in prompt_keys:
        target_sections= guess_sections(question, history_summary)
    if 'user_info_update' in prompt_keys:
        await update_cutsomer_info(question, customer_info)
    if 'vector_db_query' in prompt_keys:
        query_sentence= await generate_query_sentence(question, customer_info, history_summary)
    top_cars = []
    car_json_data: Dict[str, Any] | None = None
    car_info = []

    
    raw_hits = search_raw_hits(query_sentence, sections=target_sections, k_per_section=3)
    grouped = aggregate_by_car(raw_hits)
    top_cars = select_top_cars(grouped, top_n=3)
    car_info = "\n\n".join(
        make_car_context(car_id, info) for car_id, info in top_cars
    )
    print("\n=== Retrieved Top Vehicle Candidates ===")
    for car_id, info in top_cars:
        print(f"car_id={car_id}, score={info['score']:.4f}, chunks={len(info['chunks'])}")

    # 검색 결과가 없으면 이후 단계 중단
    if not top_cars:
        print("추천할 차량이 없습니다.")
        return

    # TODO
    # 설득 단계 대비: car_json_data 로드 (guess_sections도 실행된경우 설득 프롬프트에서 필요한 섹션 raw만 추출) 또는 vector_db_query도 실행된경우 필요한 섹션 raw만 추출) 
    json_filenames = []
    for car_id, car_info in range(top_cars):
        print("\n=== JSON 파일 찾기 ===")
        car_data_dir = Path(PROJECT_ROOT) / "chatbot" / "car_data"
        json_filename = find_json_file_for_car_id(car_id, car_data_dir)

        if json_filename:
            print(f"  → {car_id}: {json_filename}")
            try:
                car_json_data = load_car_json_data(json_filename)
                if car_json_data:
                    print("  → JSON 데이터 로드 완료")
                    json_filenames.append(json_filename)
                else:
                    print("  → JSON 데이터 로드 실패")
            except Exception as e:
                print(f"  → JSON 로드 중 오류: {e}")
                car_json_data = None
        else:
            print(f"  → {car_id}: 파일을 찾을 수 없습니다")
    
    return {target_sections: target_sections, car_json_data: car_json_data, }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Car-level RAG Search")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    # 대화 히스토리 로드
    history_file = Path(PROJECT_ROOT) / "chatbot" / "conversation_history.json"
    conversation_history = load_conversation_history(history_file)
        # 대화 히스토리 요약
    history_summary = ""
    if conversation_history:
        recent_history = conversation_history[-5:]  # 최근 5개만
        history_summary = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:100]}"
            for msg in recent_history
        ])

    print("\n=== 사전 작업 선택 ===")

    customer_info_file = Path(PROJECT_ROOT) / "chatbot" / "customer_info.json"
    customer_info: Dict[str, Any] = load_cutsomer_info(customer_info_file) 
    prompt_keys = classify_pre_task(
        args.question,
        history_summary,
        customer_info,
    )
    print(f"선택된 작업: {prompt_keys}")

    if prompt_keys is not None:
        pre_result = asyncio.run(run_pre_prompt(prompt_keys, args.question, customer_info, history_summary))
    

    print(pre_result)
        
    
    # view_function_list = ["<rotate:target>", "<zoom-in>", "<zoom-out>"]
    # print("*******************************", car_info, "***********************")
    # # LLM 답변 생성
    # answer, updated_history = answer_with_openai(
    #     question=args.question,
    #     history_summary=history_summary,
    #     conversation_history=conversation_history,
    #     customer_info=customer_info,
    #     prompt_keys=prompt_keys,
    #     car_info=car_info,
    #     view_function_list=view_function_list,
    # )
    
    # # 대화 히스토리 저장
    # save_conversation_history(updated_history, history_file)
    
    # print("\n=== Final Answer ===")
    # print(answer)


if __name__ == "__main__":
    main()
