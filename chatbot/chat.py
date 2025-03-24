from utils.embeddings import get_embedding
from vector_memory.vector_store import VectorStore
from thoughts.thoughts_manager import ThoughtsManager
from ideas.ideas_manager import IdeasManager

# Dummy response generator for now. Replace this with integration of a local LLM.
def generate_response(user_input: str, context: str) -> str:
    return f"Your input: {user_input}\n\nContext from your thoughts/ideas:\n{context}"

def main():
    # For the 'all-MiniLM-L6-v2' model, the embedding dimension is 384.
    dimension = 384
    vector_store = VectorStore(dimension)
    thoughts_manager = ThoughtsManager(vector_store)
    ideas_manager = IdeasManager(vector_store)
    
    print("Personal AI Chatbot (local-only). Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        # Ask if the user wants to save this input as a thought.
        save_choice = input("Save this as a thought? (y/n): ")
        if save_choice.lower() == "y":
            thoughts_manager.add_thought(user_input)
        
        # Retrieve context from past thoughts and ideas.
        related_thoughts = thoughts_manager.search_thoughts(user_input)
        related_ideas = ideas_manager.search_ideas(user_input)
        
        context = ""
        if related_thoughts:
            context += "Thoughts:\n" + "\n".join([t["text"] for t in related_thoughts]) + "\n"
        if related_ideas:
            context += "Ideas:\n" + "\n".join([i["text"] for i in related_ideas]) + "\n"
        
        response = generate_response(user_input, context)
        print("AI:", response)
        
        # Optionally, add a mechanism to log the conversation further or save as an idea.

if __name__ == "__main__":
    main()