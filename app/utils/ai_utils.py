import os
from venv import logger
from llama_index.core import (
    Settings
)
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from utils.configs import prompt


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

def setup_models():
    """Initialize embedding and LLM models"""
    logger.info("Setting up Gemini models...")
    
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
        
    embed_model = GeminiEmbedding(
        model_name="models/embedding-001", 
        api_key=GOOGLE_API_KEY
    )
    
    llm = Gemini(
        model="models/gemini-1.5-pro",
        api_key=GOOGLE_API_KEY, 
        temperature=0.1,  # Slightly higher for semantic search
        max_tokens=9000,
        system_prompt=prompt
    )
    
    # Set global settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = 1024  # Increased from 512
    Settings.chunk_overlap = 100  # Increased from 50
    
    logger.info("✅ Models configured successfully")


