from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import traceback

def load_api_key():      # uses python_dotenv to load  huggingface api key from .env file
    """Load Hugging Face API key from environment variables."""
    print("✅ Loading API Key...")
    load_dotenv()
    api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    print("API Key Loaded:", "Yes" if api_key else "No")
    return api_key

def initialize_llm(api_key: str):
    print("🚀 Initializing Hugging Face model...")
    try:
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
            huggingfacehub_api_token=api_key,
            temperature=0.0,
            max_new_tokens=150,  # REDUCE DRAMATICALLY from 4000 to 150
            timeout=30,  # Add timeout
            wait_for_model=True,  # Don't queue if busy
        )
        model = ChatHuggingFace(llm=llm_endpoint)
        return model
    except Exception as e:
        print("❌ Error occurred during LLM initialization:", str(e))
        traceback.print_exc()
        return None

def main():
    api_key = load_api_key()    # calls load_api_key function
    if not api_key:
        print("❌ API key not found. Exiting.")
        return

    model = initialize_llm(api_key)                   #calls function to instantiate the model and returns an instantiated model
    if model:
        print("✅ Model is ready to use.")
    else:
        print("❌ Failed to initialize model.")



if __name__ == "__main__":
    main()
