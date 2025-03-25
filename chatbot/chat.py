import sys
import os

# Append the project root to sys.path so that modules can be imported properly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.embeddings import get_embedding
from vector_memory.vector_store import VectorStore
from thoughts.thoughts_manager import ThoughtsManager
from ideas.ideas_manager import IdeasManager

# Import libraries for DeepSeek integration
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set your DeepSeek model identifier or local path.
model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load the DeepSeek model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.eval()  # Set the model to evaluation mode.

# Updated response generator using the DeepSeek model.
def generate_response(user_input: str, context: str) -> str:
    # Combine user input and context into a prompt.
    prompt = f"User: {user_input}\nContext: {context}\nAI:"
    # Use the tokenizer with return_tensors to get both input_ids and attention_mask.
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=256,
        do_sample=True,
        top_p=0.95,
        top_k=50
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

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
        
        # Generate and display the response using the DeepSeek model.
        response = generate_response(user_input, context)
        print("AI:", response)

if __name__ == "__main__":
    main()