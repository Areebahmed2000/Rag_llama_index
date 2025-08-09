from pydantic_models import ChatResponse, QueryRequest, UploadResponse
from utils.funs import save_uploaded_file
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os
from pathlib import Path
import tempfile
from typing import List
from rag_system import AgenticRAGSystem
from utils.configs import html
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG Application",
    description="An intelligent document Q&A system with conversation memory and citations",
    version="1.0.0"
)

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store RAG system instance
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    try:
        # Check if required environment variables are set
        required_vars = ["GOOGLE_API_KEY", "WEAVIATE_URL", "WEAVIATE_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        rag_system = AgenticRAGSystem()
        print("✅ Agentic RAG system initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {str(e)}")
        raise




@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface"""
    html_content = html
    return HTMLResponse(content=html_content)

@app.post("/upload_documents/", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process PDF or CSV documents"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    temp_dir = Path(tempfile.gettempdir())
    file_paths = {}
    processed_files = []
    
    try:
        for uploaded_file in files:
            # Validate file type
            allowed_extensions = ['.pdf', '.csv', '.xlsx', '.xls']
            file_extension = Path(uploaded_file.filename).suffix.lower()
            
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_extension}. Allowed types: {allowed_extensions}"
                )
            
            file_location = save_uploaded_file(uploaded_file, temp_dir)
            file_paths[uploaded_file.filename] = str(file_location)
            processed_files.append(uploaded_file.filename)
        
        # Process documents
        doc_count, node_count = rag_system.process_documents(file_paths)
        
        # Clean up temporary files
        for file_path in file_paths.values():
            try:
                os.unlink(file_path)
            except:
                pass
        
        return UploadResponse(
            message=f"Successfully processed {doc_count} documents into {node_count} chunks!",
            document_count=doc_count,
            node_count=node_count,
            files_processed=processed_files
        )
        
    except Exception as e:
        # Clean up temporary files in case of error
        for file_path in file_paths.values():
            try:
                os.unlink(file_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_documents(query: QueryRequest):
    """Chat with documents using conversation memory"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = rag_system.chat(query.question, use_agent=query.use_agent)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            conversation_id=result["conversation_id"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/ask_question/")
async def ask_question(query: QueryRequest):
    """Direct question endpoint (backward compatibility)"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = rag_system.query_with_citations(query.question)
        return {
            "question": query.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/conversation_history/")
async def get_conversation_history():
    """Get conversation history"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        history = rag_system.get_conversation_history()
        return {"conversation_history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation history: {str(e)}")

@app.post("/clear_conversation/")
async def clear_conversation():
    """Clear conversation history and memory"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        rag_system.clear_conversation_history()
        return {"message": "Conversation history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing conversation: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_system_initialized": rag_system is not None,
        "has_documents": rag_system.index is not None if rag_system else False
    }

@app.get("/system_info")
async def get_system_info():
    """Get system information"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    return {
        "system_ready": rag_system.index is not None,
        "has_chat_engine": rag_system.chat_engine is not None,
        "has_agent": rag_system.agent is not None,
        "conversation_length": len(rag_system.conversation_history),
        "exact_qa_pairs": len(rag_system.exact_qa_pairs),
        "supported_formats": [".pdf", ".csv", ".xlsx", ".xls"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)