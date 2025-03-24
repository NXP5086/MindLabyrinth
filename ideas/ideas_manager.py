import json
import os
from utils.embeddings import get_embedding
from vector_memory.vector_store import VectorStore

IDEAS_FILE = "ideas.json"

class IdeasManager:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        if os.path.exists(IDEAS_FILE):
            with open(IDEAS_FILE, "r") as f:
                self.ideas = json.load(f)
        else:
            self.ideas = []

    def add_idea(self, text: str):
        idea = {
            "type": "idea",
            "text": text
        }
        self.ideas.append(idea)
        embedding = get_embedding(text)
        self.vector_store.add_item(embedding, idea)
        self.save()

    def save(self):
        with open(IDEAS_FILE, "w") as f:
            json.dump(self.ideas, f, indent=2)

    def search_ideas(self, query: str, k: int = 5):
        embedding = get_embedding(query)
        return self.vector_store.search(embedding, k)