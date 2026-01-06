"""
중고차 챗봇 프롬프트 레지스트리

각 대화 단계별 프롬프트를 정의합니다:
- information_gathering: 정보 수집 단계 (성별, 연령, 구매목적, 예산만 수집 / 한 번에 두개씩)
- recommendation: 차량 추천 단계 (이번 요청에서는 크게 신경 X)
- persuasion: 고객 설득 단계 (customer_info 기반 + car_info 근거 + 3D 뷰 조작 포함)

그 외 유틸리티 프롬프트:
- section_classification: 섹션 분류용
- stage_classification: 단계 분류용
- customer_info_update: 정보 수집 답변에서 customer_info 업데이트용(구조화)
"""

import textwrap


# -----------------------------------------------------------------------------
# Customer Info Schema (수집 대상 4개만)
# -----------------------------------------------------------------------------
# customer_info 예시(저장 형태):
# {
#   "gender": "남|여|기타|모름",
#   "age_group": "10대|20대|30대|40대|50대|60대+|모름",
#   "purchase_purpose": ["출퇴근", "패밀리카", "레저/캠핑", "골프", "업무용", "장거리", "세컨카", "기타"],
#   "budget_krw": {"min": 0, "max": 0, "note": "현금/할부/리스 등 옵션"}
# }


PROMPT_REGISTRY = {
    "guess_sections": {
        "system": textwrap.dedent("""
            "당신은 질문에 필요한 차량 정보 섹션을 판단해 주는 어시스턴트입니다."
        """).strip(),

        "user_template":textwrap.dedent("""
            아래 질문과 대화 기록을 통해 고객의 니즈를 추론하여 차량의 어떤 정보(섹션)가 필요할지 빠르게 판단.

            가능한 섹션 이름:
            summary, basic_info, special_usage_history, special_accident_history,
            registration_change_history, insurance_accident_history, options,
            seller_info, inspection_record

            user_query: {question}
            이전 대화 요약:
            {history_summary}

            출력 형식: 쉼표로 구분된 섹션 이름.
        """)
    },

    "user_info_update": {
        "system": textwrap.dedent("""
            당신은 대화에서 customer_info를 갱신하는 정보 추출기입니다.
            customer_info는 아래 4개 필드만 관리합니다:
            - gender: "남" | "여" | "기타" | "모름"
            - age_group: "10대" | "20대" | "30대" | "40대" | "50대" | "60대+" | "모름"
            - purchase_purpose: 문자열 리스트(예: ["출퇴근","패밀리카","골프"]).
            - budget_krw: 만원단위 숫자

            규칙:
            - 사용자가 명시한 것만 채워라. 추정 금지.
            - 값이 불명확하면 해당 필드는 변경하지 마라.
            - purchase_purpose는 중복 없이 누적 가능.
            - 결과는 반드시 JSON만 출력하라.
        """).strip(),

        "user_template": textwrap.dedent("""
            고객 정보:{customer_info}
            user_query:{question}

            출력(JSON only):
            {{"gender": "...", "age_group": "...", "purchase_purpose": [...], "budget_krw": "..."}}
        """).strip()
    },

   "vector_db_query": {
        "system": textwrap.dedent("""
            당신은 차량을 찾아주기 위해 벡터 데이터베이스에 쿼리할 적절한 문장을 작성하는 어시스턴트 입니다.
            유저의 질문은 특정 차량 모델명 혹은 원하는 차량의 특성이 될 수 있습니다.

            예시) 
            1. 유저가 특정 차량 모델명을 말한 경우의 결과값: 2020년식 벤틀리 컨티넨탈 GT V8
            2. 유저가 원하는 차량의 특성을 말한 경우의 결과값: 1억 이하, 흰색, 통풍 시트가 구비되어 있는 무사고 SUV.
            
        """).strip(),

        "user_template": textwrap.dedent("""
            고객 정보:{customer_info}
            user_query:{question}
            이전 대화 요약:
            {history_summary}
            출력: 자연어 문장
        """).strip()
    }
}


# ---------------------------------------------------------------------------
# Utility Prompts
# ---------------------------------------------------------------------------


# 단계 분류 프롬프트 (정보수집 우선 로직 강화: 4개 미완이면 information_gathering)

PRE_STAGE_CLASSIFICATION_SYSTEM = "당신은 대화에 필요한 사전정보를 파악하는 AI입니다. 정확하고 빠르게 판단하세요."

PRE_STAGE_CLASSIFICATION_USER_TEMPLATE = textwrap.dedent("""
    다음 대화에서 어떤 프롬프트가 필요한지 판단하세요.(복수 선택 가능)

    가능한 프롬프트:
    - guess_sections
    - user_info_update
    - vector_db_query

    user_query: {question}

    이전 대화 요약:
    {history_summary}

    customer_info(현재):
    {customer_info}

    판단 규칙:
    1) 차량의 정보중에 고객이 궁금해하거나, 고객에게 제공할 것을 선정해야 할 시 → guess_sections
    2) 차량 추천에 필요한 [성별, 나이, 예산, 구매목적] 중 하나 이상이 고객의 현재 발화에 포함되어 있을 시 → user_info_update
    3) 차량을 추천하거나 차량 정보를 제공해야할 시 → vector_db_query

    출력 형식: 쉼표로 구분된 프롬프트 키 (guess_sections 또는 user_info_update,vector_db_query)
""").strip()