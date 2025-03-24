import json
import os
from utils.embeddings import get_embedding
from vector_memory.vector_store import VectorStore

THOUGHTS_FILE = "thoughts.json"

class ThoughtsManager:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        if os.path.exists(THOUGHTS_FILE):
            with open(THOUGHTS_FILE, "r") as f:
                self.thoughts = json.load(f)
        else:
            self.thoughts = []

    def add_thought(self, text: str):
        thought = {
            "type": "thought",
            "text": text
        }
        self.thoughts.append(thought)
        embedding = get_embedding(text)
        self.vector_store.add_item(embedding, thought)
        self.save()

    def save(self):
        with open(THOUGHTS_FILE, "w") as f:
            json.dump(self.thoughts, f, indent=2)

    def search_thoughts(self, query: str, k: int = 5):
        embedding = get_embedding(query)
        return self.vector_store.search(embedding, k)