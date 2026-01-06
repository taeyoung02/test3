# RAG 챗봇
1. **임베딩 생성**  
   ```bash
   python chatbot/NL_to_VEC.py
   ```  
   - `.env`에 `OPENAI_API_KEY` 필요.  
   - 모든 `car_info*.json` → `car_info*_vector.json` 생성.

2. **Qdrant 업서트**  
   - `build_vectordb2.py`에서 `QdrantClient`를 실제 서버로 설정 후 `recreate_collection` 사용.  
   ```python
   client = QdrantClient(url="http://localhost:6333", api_key="...")
   client.recreate_collection("used_car_info", ...)
   ```  
   - 실행: `python chatbot/build_vectordb2.py`

3. **질문 실행**  
   ```bash
   python chatbot/qdrant_openai_query.py \
     --question "열선 시트 있는 SUV 추천해줘" \
   ```  
   - 질문 임베딩 → Qdrant 검색 → 차량별 점수 집계 → LLM 답변.  
