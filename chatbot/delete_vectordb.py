from qdrant_client import QdrantClient
import os
client = QdrantClient(url="http://localhost:6333", api_key=os.getenv("QDRANT_API_KEY"))
client.delete_collection("used_car_info")

print("Vector database deleted successfully")