import sys
import os
import numpy as np

# Append the project root to sys.path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from vector_memory.vector_store import VectorStore
from ideas.ideas_manager import IdeasManager

def main():
    # Set the embedding dimension for "all-MiniLM-L6-v2" (384 is common for this model)
    dimension = 384
    
    # Optionally, clear any existing ideas file for a fresh test
    ideas_file = "ideas.json"
    if os.path.exists(ideas_file):
        os.remove(ideas_file)
    
    # Initialize the shared vector store and IdeasManager.
    store = VectorStore(dimension)
    ideas_manager = IdeasManager(store)
    
    # Add some ideas.
    print("Adding ideas...")
    ideas_manager.add_idea("Build a personal AI assistant.")
    ideas_manager.add_idea("Explore local language models for privacy.")
    ideas_manager.add_idea("Improve the user interface for better UX.")
    
    # Search for ideas that relate to "personal AI"
    query = "personal AI"
    print(f"\nSearching for ideas related to: '{query}'")
    results = ideas_manager.search_ideas(query, k=3)
    
    # Display the search results.
    print("\nSearch results:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()