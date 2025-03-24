from sentence_transformers import SentenceTransformer

# Load a local model (downloaded beforehand, so nothing leaves your machine)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str):
    return model.encode(text)