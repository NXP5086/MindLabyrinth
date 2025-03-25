import sys
import os

# Append the project root to sys.path so that modules can be imported properly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from vector_memory.vector_store import VectorStore
from thoughts.thoughts_manager import ThoughtsManager

def main():
    # Set the embedding dimension (e.g., 384 for "all-MiniLM-L6-v2")
    dimension = 384

    # Optionally, remove an existing thoughts.json file for a clean test.
    thoughts_file = "thoughts.json"
    if os.path.exists(thoughts_file):
        os.remove(thoughts_file)

    # Initialize the vector store and ThoughtsManager.
    vector_store = VectorStore(dimension)
    tm = ThoughtsManager(vector_store)

    # Add several thoughts.
    print("Adding thoughts...")
    tm.add_thought("I love programming in Python.")
    tm.add_thought("Today I learned about vector databases.")
    tm.add_thought("I'm excited to build AI systems.")
    tm.add_thought("The weather is quite pleasant today.")
    tm.add_thought("Reviewing code helps me improve.")

    # Search for thoughts similar to a query.
    query = "Python programming"
    print(f"\nSearching for thoughts related to: '{query}'")
    results = tm.search_thoughts(query, k=5)
    
    # Display the search results.
    print("\nSearch results:")
    for result in results:
        print("  ", result)

if __name__ == "__main__":
    main()