---
config:
  layout: fixed
---
flowchart TB
    U["사용자 (웹/채팅 UI)"] -- 텍스트 질의 --> GW["API Gateway / Chat Server"]
    GW -- 질문, 기존 대화 기록, customer_info --> Orchestrator["Chat Orchestrator"]
    StageClassifier["StageClassifier"] -- 단계: 
    정보탐색/추천/설득 --> Orchestrator
    Orchestrator -- 정보탐색 단계 --> InfoAgent["정보탐색 에이전트"]
    Orchestrator -- 추천 단계 --> RecAgent["추천 에이전트"]
    Orchestrator -- 설득 단계 --> PersuasionAgent["설득 에이전트"]
    RecAgent -- 벡터 검색 쿼리 --> Qdrant[("Qdrant Vector DB")]
    Qdrant -- 섹션별 후보 문서 --> RecAgent
    RecAgent -- 섹션 요약 + 후보 차량 리스트 --> Orchestrator
    PersuasionAgent -- 필요한 차량 정보 결정 --> SecClass["의도 추출(LLM)"]
    PersuasionAgent -- car_id --> CarJson["Car JSON"]
    CarJson -- 필요한 섹션 추출 --> PersuasionAgent
    PersuasionAgent -- sales_knowledge 선택 --> SalesK["Sales Knowledge(RDB)"]
    PersuasionAgent -- view function 선택 --> ViewFn["View functions"]
    Orchestrator -- 시스템+단계별 프롬프트 --> LLM["OpenAI Chat API"]
    LLM -- 시나리오/답변 텍스트 (+3D 태그) --> GW
    GW -- 응답 메시지/TTS + 3D control --> U
    InfoAgent --> n1["User Info(RDB)"]
    n1 --> RecAgent & PersuasionAgent

     InfoAgent:::agent
     RecAgent:::agent
     PersuasionAgent:::agent
    classDef agent fill:#ffe5b4,stroke:#f39c12,stroke-width:2px,color:#000