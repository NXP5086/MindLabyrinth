import faiss
import numpy as np
import os
import pickle

class VectorStore:
    def __init__(self, dimension: int, index_file: str = "vector_index.pkl", data_file: str = "vector_data.pkl"):
        self.dimension = dimension
        self.index_file = index_file
        self.data_file = data_file
        if os.path.exists(self.index_file) and os.path.exists(self.data_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.data_file, "rb") as f:
                self.data = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.data = []

    def add_item(self, embedding: np.ndarray, item: dict):
        embedding = np.array([embedding]).astype('float32')
        self.index.add(embedding)
        self.data.append(item)
        self.save()

    def search(self, query_embedding: np.ndarray, k: int = 5):
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx in indices[0]:
            if idx < len(self.data):
                results.append(self.data[idx])
        return results

    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.data_file, "wb") as f:
            pickle.dump(self.data, f)