from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models as rest
import json
from pathlib import Path
from typing import Any, Iterable

import uuid


#  도커로 qdrant 서버 띄우고 나서 실행
client = QdrantClient(host="localhost", port=6333)  

# 또는 QdrantClient(":memory:") 모든 데이터는 RAM에만 올라가고, 프로세스 끝나면 사라짐

# 🔹 2. 컬렉션 생성 (그대로 사용)
client.create_collection(
    collection_name="used_car_info",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# 🔹 3. 로컬 JSON 로드 (그대로)
def iter_car_data_vector_files(datapath: Path) -> Iterable[Path]:
    for path in sorted(datapath.glob("car_info*_vector.json")):
        yield json.loads(path.read_text())



points = []
DATA_DIR = Path("chatbot/car_data")
for records in iter_car_data_vector_files(DATA_DIR):
    for rec in records:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, rec["id"]))
        points.append(
            rest.PointStruct(
                id=point_id,
                vector=rec["vector"],
                payload=rec["payload"],
            )
        )

# 🔹 5. 메모리 Qdrant에 upsert
client.upsert(collection_name="used_car_info", points=points)




# ✅ 업서트 확인
total = client.count(collection_name="used_car_info", exact=True)
print("총 포인트:", total.count)