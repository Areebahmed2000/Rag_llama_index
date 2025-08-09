
from utils.ai_utils import setup_models
import weaviate
from weaviate.auth import Auth
import os
from typing import List, Dict, Any, Optional
import difflib
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SimilarityPostprocessor
import logging
from doc_processor import DocumentProcessor
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")



class AgenticRAGSystem:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.index = None
        self.query_engine = None
        self.chat_engine = None
        self.agent = None
        self.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        self.vector_store = None
        self.weaviate_client = None
        self.conversation_history = []
        
        # Initialize storage for exact Q&A pairs
        self.exact_qa_pairs = {}  # For exact matching
        
        # Initialize models and setup
        setup_models()
        self.setup_weaviate()

    def setup_weaviate(self):
        """Setup Weaviate vector database connection"""
        try:
            if not WEAVIATE_URL or not WEAVIATE_API_KEY:
                raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY environment variables are required")
                
            self.weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
            )
            
            if not self.weaviate_client.is_ready():
                raise Exception("Weaviate client not ready")
                
            logger.info("✅ Successfully connected to Weaviate")
            self.setup_collection()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Weaviate: {str(e)}")
            raise

    def setup_collection(self):
        """Setup Weaviate collection for document storage"""
        collection_name = "Documents"
        try:
            # Delete existing collection if it exists
            if self.weaviate_client.collections.exists(collection_name):
                self.weaviate_client.collections.delete(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")

            # Create new collection
            self.weaviate_client.collections.create(
                name=collection_name,
                properties=[
                    weaviate.classes.config.Property(
                        name="content", 
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="file_name", 
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="page_number", 
                        data_type=weaviate.classes.config.DataType.INT
                    ),
                    weaviate.classes.config.Property(
                        name="document_type", 
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                ],
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none()
            )
            
            self.vector_store = WeaviateVectorStore(
                weaviate_client=self.weaviate_client,
                index_name=collection_name,
                text_key="content"
            )
            
            logger.info(f"✅ Created collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Error setting up collection: {str(e)}")
            raise

    def process_documents(self, file_paths: Dict[str, str]):
        """Process uploaded documents and build index"""
        all_documents = []
        
        # Clear previous exact Q&A pairs
        self.exact_qa_pairs = {}
        
        for filename, filepath in file_paths.items():
            logger.info(f"Processing file: {filename}")
            
            try:
                if filename.lower().endswith('.pdf'):
                    docs = self.doc_processor.extract_text_from_pdf(filepath, filename)
                    all_documents.extend(docs)
                elif filename.lower().endswith('.csv'):
                    docs = self.doc_processor.load_qa_from_csv(filepath)
                    all_documents.extend(docs)
                    # Store Q&A pairs for exact matching
                    self._store_exact_qa_pairs(docs)
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    docs = self.doc_processor.load_qa_from_excel(filepath)
                    all_documents.extend(docs)
                    # Store Q&A pairs for exact matching
                    self._store_exact_qa_pairs(docs)
                else:
                    logger.warning(f"Skipping unsupported file type: {filename}")
                    
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
                
        if not all_documents:
            raise ValueError("No documents could be processed successfully")

        logger.info(f"Loaded {len(all_documents)} documents")
        logger.info(f"Stored {len(self.exact_qa_pairs)} exact Q&A pairs")
        logger.info(f"all docsa re {all_documents}")
        
        # Create nodes from documents
        nodes = self.doc_processor.create_nodes_with_metadata(all_documents)
        logger.info(f"Created {len(nodes)} nodes")
        
        # Build the index and engines
        self.build_index_and_engines(nodes)
        
        return len(all_documents), len(nodes)

    def _store_exact_qa_pairs(self, documents):
        """Store Q&A pairs for exact matching with length limits"""
        for doc in documents:
            if doc.metadata.get('type') == 'qa_pair':
                original_q = doc.metadata.get('original_question', '').strip()
                original_a = doc.metadata.get('original_answer', '').strip()
                
                if original_q and original_a:
                    # Ensure reasonable lengths for storage
                    if len(original_q) > 300:
                        original_q = original_q[:300] + "..."
                    if len(original_a) > 1000:
                        original_a = original_a[:1000] + "..."
                        
                    question_key = original_q.lower()
                    self.exact_qa_pairs[question_key] = {
                        'original_question': original_q,
                        'original_answer': original_a,
                        'source': doc.metadata.get('source', ''),
                        'file_name': doc.metadata.get('file_name', ''),
                        'page_number': doc.metadata.get('page_number', 1),
                        'document_type': doc.metadata.get('document_type', ''),
                        'sheet_name': doc.metadata.get('sheet_name', '')[:20] if doc.metadata.get('sheet_name') else ''
                    }

    def find_exact_match(self, question: str) -> Optional[Dict[str, Any]]:
        """Find exact question match in stored Q&A pairs"""
        question_clean = question.strip().lower()
        
        # Direct exact match
        if question_clean in self.exact_qa_pairs:
            match = self.exact_qa_pairs[question_clean]
            logger.info(f"🎯 Found EXACT match for: '{question[:50]}...'")
            return {
                'answer': match['original_answer'],
                'source': match,
                'match_type': 'exact'
            }
        
        # Very high similarity matching (95%+ similarity)
        best_match = None
        best_ratio = 0
        
        for stored_question, qa_data in self.exact_qa_pairs.items():
            # Use sequence matching for similarity
            ratio = difflib.SequenceMatcher(None, question_clean, stored_question).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = qa_data
        
        # Only accept very high similarity (95%+) for exact match category
        if best_match and best_ratio >= 0.95:
            logger.info(f"🔍 Found FUZZY EXACT match (similarity: {best_ratio:.3f}) for: '{question[:50]}...'")
            return {
                'answer': best_match['original_answer'],
                'source': best_match,
                'match_type': 'fuzzy_exact',
                'similarity': best_ratio
            }
        
        logger.info(f"❌ No exact match found for: '{question[:50]}...'. Best similarity: {best_ratio:.3f}")
        return None

    def build_index_and_engines(self, nodes: List[TextNode]):
        """Build vector index and create query/chat engines"""
        try:
            # Build vector index
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex(nodes, storage_context=storage_context)
            
            # Create query engine for semantic search
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact",
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.7)  # Moderate threshold for semantic
                ]
            )
            
            # Create chat engine for conversation memory
            self.chat_engine = self.index.as_chat_engine(
                chat_mode="condense_plus_context",
                memory=self.chat_memory,
                verbose=True,
                similarity_top_k=5,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.7)
                ]
            )
            
            # Create agentic tools
            from llama_index.core.tools import QueryEngineTool, ToolMetadata
            
            query_tool = QueryEngineTool(
                query_engine=self.query_engine,
                metadata=ToolMetadata(
                    name="document_search",
                    description="Search through uploaded documents to find relevant information and citations"
                )
            )
            
            # Create simple agent (fallback if ReAct agent fails)
            try:
                from llama_index.core.agent import ReActAgent
                self.agent = ReActAgent.from_tools(
                    [query_tool],
                    llm=self.llm,
                    verbose=True,
                    max_iterations=3
                )
            except Exception as e:
                logger.warning(f"Could not create ReAct agent, using simple query engine: {e}")
                self.agent = None
            
            logger.info("✅ Successfully built index, query engine, chat engine, and agent")
            
        except Exception as e:
            logger.error(f"❌ Error building engines: {str(e)}")
            raise

    def chat(self, message: str, use_agent: bool = True) -> Dict[str, Any]:
        """2-Priority Hybrid Search: 1) Exact Match 2) Single Best Semantic Match"""
        if not self.chat_engine:
            raise ValueError("System not initialized. Please upload documents first.")
        
        try:
            # Store user message
            self.conversation_history.append({"role": "user", "content": message})
            
            # PRIORITY 1: Try exact question matching first
            exact_match = self.find_exact_match(message)
            
            if exact_match:
                response_text = exact_match['answer']
                sources = [{
                    'file_name': exact_match['source']['file_name'],
                    'page_number': exact_match['source']['page_number'],
                    'document_type': 'exact_match',
                    'source': exact_match['source']['source'],
                    'match_type': exact_match['match_type'],
                    'similarity_score': exact_match.get('similarity', 1.0),
                    'original_question': exact_match['source']['original_question'],
                    'original_answer': exact_match['source']['original_answer']
                }]
                
                logger.info(f"✅ PRIORITY 1: Exact match found ({exact_match['match_type']})")
                
            else:
                # PRIORITY 2: Single best semantic search (top_k=1)
                logger.info("PRIORITY 2: Using semantic search with top_k=1...")
                
                # Create a single-result query engine for this specific query
                single_result_engine = self.index.as_query_engine(
                    similarity_top_k=1,  # Only get the single best match
                    response_mode="compact",
                    node_postprocessors=[
                        SimilarityPostprocessor(similarity_cutoff=0.75)  # High threshold
                    ]
                )
                
                response = single_result_engine.query(message)
                response_text = str(response)
                sources = self._extract_sources_from_response(response)
                
                # Check if the single semantic result is good enough
                if sources and len(sources) > 0:
                    best_source = sources[0]
                    similarity_score = best_source.get('similarity_score', 0)
                    
                    logger.info(f"Best semantic match similarity: {similarity_score}")
                    
                    # If similarity is high enough, use it
                    if similarity_score >= 0.75:  # High similarity threshold
                        logger.info("✅ PRIORITY 2: High-similarity semantic match found")
                        
                        # If it's from Q&A pairs, use the exact answer
                        if 'original_answer' in best_source:
                            response_text = best_source['original_answer']
                            logger.info("Using exact answer from Q&A pair")
                        
                        # Mark as semantic match
                        for source in sources:
                            source['match_type'] = 'semantic_high'
                            
                    else:
                        # Similarity too low
                        logger.info(f"❌ PRIORITY 2: Similarity too low ({similarity_score:.2f} < 0.75)")
                        response_text = "No information available in our RAG system."
                        sources = []
                else:
                    # No semantic matches found
                    logger.info("❌ PRIORITY 2: No semantic matches found")
                    response_text = "No information available in our RAG system."
                    sources = []
            
            # Store assistant response
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return {
                "answer": response_text,
                "sources": sources,
                "conversation_id": len(self.conversation_history) // 2
            }
            
        except Exception as e:
            logger.error(f"❌ Error during hybrid search: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "sources": [],
                "conversation_id": len(self.conversation_history) // 2
            }

    def _extract_sources_from_response(self, response) -> List[Dict[str, Any]]:
        """Extract source citations from response with original answer extraction"""
        sources = []
        
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                source_info = {
                    "file_name": node.metadata.get("file_name", "Unknown"),
                    "page_number": node.metadata.get("page_number", "Unknown"),
                    "document_type": node.metadata.get("document_type", "Unknown"),
                    "source": node.metadata.get("source", "Unknown"),
                    "type": node.metadata.get("type", "Unknown"),
                    "similarity_score": getattr(node, 'score', None)
                }
                
                # Add original Q&A if available
                if 'original_question' in node.metadata:
                    source_info['original_question'] = node.metadata['original_question']
                if 'original_answer' in node.metadata:
                    source_info['original_answer'] = node.metadata['original_answer']
                    
                sources.append(source_info)
        
        return sources

    def _has_relevant_content(self, response_text: str) -> bool:
        """Check if response contains relevant information or should be marked as 'no answer found'"""
        response_lower = response_text.lower().strip()
        
        # Direct "no answer" indicators
        no_answer_phrases = [
            "no information available",
            "no answer found",
            "i don't have information",
            "i cannot find",
            "cannot find",
            "no information",
            "not mentioned",
            "not provided",
            "not available",
            "i don't see",
            "there is no information",
            "based on the provided context, i cannot",
            "the provided documents do not contain",
            "no relevant information"
        ]
        
        # Check for direct no-answer indicators
        for phrase in no_answer_phrases:
            if phrase in response_lower:
                return False
        
        # Check for very short or generic responses that likely indicate no real answer
        if len(response_lower) < 20:  # Very short responses are suspicious
            return False
            
        # Check for responses that are just apologetic without content
        apologetic_only = [
            "i apologize",
            "sorry",
            "i'm sorry"
        ]
        
        if any(phrase in response_lower for phrase in apologetic_only) and len(response_lower) < 100:
            return False
            
        return True

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()

    def clear_conversation_history(self):
        """Clear conversation history and memory"""
        self.conversation_history = []
        if self.chat_memory:
            self.chat_memory.reset()
        logger.info("Conversation history cleared")

    def query_with_citations(self, question: str) -> Dict[str, Any]:
        """Direct query method for backward compatibility"""
        return self.chat(question, use_agent=False)