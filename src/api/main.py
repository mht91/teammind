"""
FastAPI Backend for TeamMind with Groq Integration
Updated with proper data folder structure
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import tempfile
from loguru import logger
from dotenv import load_dotenv
import json
import uuid

# Load environment variables
load_dotenv()

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Import components
from ..ingestion.audio_processor import AudioProcessor
from ..ingestion.document_processor import DocumentProcessor
from ..ingestion.image_processor import ImageProcessor
from ..processing.chunking import SemanticChunker
from ..processing.embeddings import MultimodalEmbedder
from ..retrieval.simple_vector_store import SimpleVectorStore
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import Reranker
from ..orchestration.query_planner import QueryPlanner
from ..orchestration.context_fusion import ContextFusion
from ..orchestration.synthesizer import Synthesizer
from ..agents import (
    SummaryAgent, DecisionAgent, RiskAgent, 
    TaskAgent, WeakSignalAgent, KnowledgeGraphAgent
)

# Initialize FastAPI app
app = FastAPI(
    title="TeamMind API", 
    version="1.0.0",
    description="Multi-Agent Meeting Intelligence System with Groq LLM"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    query: str
    filter_conditions: Optional[Dict[str, Any]] = None
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    agent_outputs: Dict[str, Any]
    metadata: Dict[str, Any]

# Global variables for components
audio_processor = None
document_processor = None
image_processor = None
chunker = None
embedder = None
vector_store = None
retriever = None
reranker = None
query_planner = None
context_fusion = None
synthesizer = None

# Agents
summary_agent = None
decision_agent = None
risk_agent = None
task_agent = None
weak_signal_agent = None
knowledge_graph_agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global audio_processor, document_processor, image_processor
    global chunker, embedder, vector_store, retriever, reranker
    global query_planner, context_fusion, synthesizer
    global summary_agent, decision_agent, risk_agent, task_agent, weak_signal_agent, knowledge_graph_agent
    
    # Get Groq credentials
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    if not groq_api_key:
        logger.warning("⚠️ GROQ_API_KEY not set in environment variables")
        logger.warning("Please set GROQ_API_KEY in .env file")
    
    logger.info("🚀 Initializing TeamMind with Groq LLM")
    
    try:
        # Initialize processors
        logger.info("📦 Initializing Audio Processor...")
        audio_processor = AudioProcessor(
            model_name=os.getenv("WHISPER_MODEL", "base")
        )
        
        logger.info("📦 Initializing Document Processor...")
        document_processor = DocumentProcessor(chunk_size=500, overlap=50)
        
        logger.info("📦 Initializing Image Processor...")
        image_processor = ImageProcessor(
            model_name=os.getenv("IMAGE_MODEL", "openai/clip-vit-base-patch32")
        )
        
        # Initialize processing pipeline
        logger.info("📦 Initializing Semantic Chunker...")
        chunker = SemanticChunker(
            model_name=os.getenv("TEXT_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        )
        
        logger.info("📦 Initializing Embedder...")
        embedder = MultimodalEmbedder(
            text_model=os.getenv("TEXT_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        )
        
        # Initialize vector store with data folder
        logger.info("📦 Initializing Vector Store...")
        vector_store = SimpleVectorStore(
            collection_name="teammind",
            vector_size=768,
            db_path="data/vector_store.db"
        )
        
        # Initialize retrieval
        logger.info("📦 Initializing Hybrid Retriever...")
        retriever = HybridRetriever(vector_store, embedder)
        
        logger.info("📦 Initializing Reranker...")
        reranker = Reranker(
            model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
        )
        
        # Initialize orchestration
        logger.info("📦 Initializing Query Planner...")
        query_planner = QueryPlanner()
        
        logger.info("📦 Initializing Context Fusion...")
        context_fusion = ContextFusion()
        
        logger.info("📦 Initializing Synthesizer...")
        synthesizer = Synthesizer(
            groq_api_key=groq_api_key,
            groq_model=groq_model
        )
        
        # Initialize agents with Groq
        logger.info("📦 Initializing Agents with Groq...")
        summary_agent = SummaryAgent(
            groq_api_key=groq_api_key,
            groq_model=groq_model
        )
        decision_agent = DecisionAgent(
            groq_api_key=groq_api_key,
            groq_model=groq_model
        )
        risk_agent = RiskAgent(
            groq_api_key=groq_api_key,
            groq_model=groq_model
        )
        task_agent = TaskAgent(
            groq_api_key=groq_api_key,
            groq_model=groq_model
        )
        weak_signal_agent = WeakSignalAgent(
            groq_api_key=groq_api_key,
            groq_model=groq_model
        )
        knowledge_graph_agent = KnowledgeGraphAgent(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            groq_api_key=groq_api_key,
            groq_model=groq_model
        )
        
        logger.info("✅ All components initialized successfully with Groq!")
        logger.info(f"📊 Vector Store: {vector_store.db_path}")
        logger.info(f"🤖 LLM: {groq_model}")
        logger.info(f"🎯 Agents: 6 specialized agents active")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """
    Upload and process a file (audio, document, image)
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        # Parse metadata
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except:
                try:
                    metadata_dict = eval(metadata)
                except:
                    metadata_dict = {}
        
        # Process based on file type
        ext = Path(file.filename).suffix.lower()
        result = None
        processed_chunks = []
        
        if ext in ['.mp3', '.wav', '.m4a', '.flac']:
            if audio_processor:
                result = audio_processor.process_audio(tmp_path, metadata_dict)
                if result and 'segments' in result:
                    for seg in result['segments']:
                        processed_chunks.append({
                            'text': seg.get('text', ''),
                            'metadata': {
                                'modality': 'audio',
                                'speaker': seg.get('speaker', 'Unknown'),
                                'timestamp': seg.get('start', 0),
                                'source_file': file.filename,
                                **metadata_dict
                            }
                        })
        elif ext in ['.pdf', '.docx', '.pptx', '.txt', '.md', '.html', '.json']:
            if document_processor:
                chunks = document_processor.process_document(tmp_path, metadata_dict)
                result = {"chunks": []}
                for chunk in chunks:
                    processed_chunks.append({
                        'text': chunk.content,
                        'metadata': {
                            'modality': 'document',
                            'page': getattr(chunk, 'page_number', None),
                            'slide': getattr(chunk, 'slide_number', None),
                            'source_file': file.filename,
                            **metadata_dict
                        }
                    })
                    result['chunks'].append(chunk.__dict__)
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            if image_processor:
                result = image_processor.process_image(tmp_path, metadata_dict)
                if result and 'description' in result:
                    processed_chunks.append({
                        'text': result['description'],
                        'metadata': {
                            'modality': 'image',
                            'description': result.get('description', ''),
                            'source_file': file.filename,
                            **metadata_dict
                        }
                    })
        
        # Store chunks in vector store
        chunks_stored = 0
        if processed_chunks and vector_store and embedder:
            vectors = []
            payloads = []
            ids = []
            
            for chunk in processed_chunks:
                text = chunk.get('text', '')
                if text and len(text.strip()) > 10:  # Only store meaningful text
                    try:
                        # Generate embedding
                        embedding = embedder.embed_text(text)[0]
                        vectors.append(embedding)
                        
                        # Prepare payload
                        payload = chunk.get('metadata', {})
                        payload['text'] = text
                        payloads.append(payload)
                        ids.append(str(uuid.uuid4()))
                    except Exception as e:
                        logger.error(f"Error embedding chunk: {e}")
            
            if vectors:
                vector_store.add_points(vectors, payloads, ids)
                chunks_stored = len(vectors)
                logger.info(f"✅ Stored {chunks_stored} chunks in vector store")
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "result": result,
            "chunks_stored": chunks_stored,
            "total_chunks_extracted": len(processed_chunks)
        }
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query using Groq-powered agents
    """
    try:
        # Step 1: Plan query
        query_plan = query_planner.plan(request.query)
        
        # Step 2: Retrieve relevant information
        try:
            retrieved = retriever.retrieve(
                query=request.query,
                filter_conditions=request.filter_conditions,
                top_k=request.top_k * 2
            )
            logger.info(f"Retrieved {len(retrieved)} chunks")
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            retrieved = []
        
        # Step 3: Rerank results
        if retrieved:
            reranked = reranker.rerank(request.query, retrieved, request.top_k)
        else:
            reranked = []
            logger.warning("No retrieved chunks to rerank")
        
        # Step 4: Fuse context
        fused_context = context_fusion.fuse(reranked, query_plan)
        
        # Step 5: Run agents
        agent_outputs = await run_agents(fused_context, query_plan)
        
        # Step 6: Generate response
        response = synthesizer.synthesize(request.query, fused_context, query_plan)
        
        # Add agent outputs to response
        response.agent_outputs = agent_outputs
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

async def run_agents(context: Any, query_plan: Any) -> Dict[str, Any]:
    """
    Run all agents in parallel
    """
    agent_outputs = {}
    
    # Convert context to dict if needed
    if hasattr(context, 'to_dict'):
        context_dict = context.to_dict()
    elif isinstance(context, dict):
        context_dict = context
    else:
        context_dict = {
            'text': getattr(context, 'text', ''),
            'source_chunks': getattr(context, 'source_chunks', []),
            'metadata': getattr(context, 'metadata', {}),
            'agent_inputs': getattr(context, 'agent_inputs', {})
        }
    
    # Get text and chunks from context
    meeting_text = context_dict.get('text', '')
    retrieved_chunks = context_dict.get('source_chunks', [])
    metadata = context_dict.get('metadata', {})
    
    # Log what we're working with
    logger.info(f"Running agents with {len(retrieved_chunks)} chunks, {len(meeting_text)} chars")
    
    # Prepare agent input
    agent_input = {
        'meeting_text': meeting_text,
        'retrieved_chunks': retrieved_chunks,
        'metadata': metadata,
        'query_plan': query_plan,
        'historical_data': []
    }
    
    # Run all agents with error handling
    
    # 1. Summary Agent
    try:
        logger.info("🔄 Running Summary Agent...")
        summary_output = summary_agent.process(agent_input)
        agent_outputs['summary'] = summary_output.__dict__ if hasattr(summary_output, '__dict__') else summary_output
        logger.info("✅ Summary Agent completed")
    except Exception as e:
        logger.error(f"❌ Summary agent error: {e}")
        agent_outputs['summary'] = {
            'error': str(e), 
            'content': 'Summary agent failed',
            'type': 'summary',
            'confidence': 0.0
        }
    
    # 2. Decision Agent
    try:
        logger.info("🔄 Running Decision Agent...")
        decision_output = decision_agent.process(agent_input)
        agent_outputs['decision'] = decision_output.__dict__ if hasattr(decision_output, '__dict__') else decision_output
        logger.info("✅ Decision Agent completed")
    except Exception as e:
        logger.error(f"❌ Decision agent error: {e}")
        agent_outputs['decision'] = {
            'error': str(e), 
            'content': 'Decision agent failed',
            'type': 'decision',
            'confidence': 0.0
        }
    
    # 3. Risk Agent
    try:
        logger.info("🔄 Running Risk Agent...")
        risk_output = risk_agent.process(agent_input)
        agent_outputs['risk'] = risk_output.__dict__ if hasattr(risk_output, '__dict__') else risk_output
        logger.info("✅ Risk Agent completed")
    except Exception as e:
        logger.error(f"❌ Risk agent error: {e}")
        agent_outputs['risk'] = {
            'error': str(e), 
            'content': 'Risk agent failed',
            'type': 'risk',
            'confidence': 0.0
        }
    
    # 4. Task Agent
    try:
        logger.info("🔄 Running Task Agent...")
        task_output = task_agent.process(agent_input)
        agent_outputs['task'] = task_output.__dict__ if hasattr(task_output, '__dict__') else task_output
        logger.info("✅ Task Agent completed")
    except Exception as e:
        logger.error(f"❌ Task agent error: {e}")
        agent_outputs['task'] = {
            'error': str(e), 
            'content': 'Task agent failed',
            'type': 'task',
            'confidence': 0.0
        }
    
    # 5. Weak Signal Agent
    try:
        logger.info("🔄 Running Weak Signal Agent...")
        signal_output = weak_signal_agent.process(agent_input)
        agent_outputs['weak_signal'] = signal_output.__dict__ if hasattr(signal_output, '__dict__') else signal_output
        logger.info("✅ Weak Signal Agent completed")
    except Exception as e:
        logger.error(f"❌ Weak signal agent error: {e}")
        agent_outputs['weak_signal'] = {
            'error': str(e), 
            'content': 'Weak signal agent failed',
            'type': 'weak_signal',
            'confidence': 0.0
        }
    
    # 6. Knowledge Graph Agent
    try:
        logger.info("🔄 Running Knowledge Graph Agent...")
        graph_output = knowledge_graph_agent.process(agent_input)
        agent_outputs['knowledge_graph'] = graph_output.__dict__ if hasattr(graph_output, '__dict__') else graph_output
        logger.info("✅ Knowledge Graph Agent completed")
    except Exception as e:
        logger.error(f"❌ Knowledge graph agent error: {e}")
        agent_outputs['knowledge_graph'] = {
            'error': str(e), 
            'content': 'Knowledge graph agent failed',
            'type': 'knowledge_graph',
            'confidence': 0.0
        }
    
    logger.info(f"✅ All agents completed. {len(agent_outputs)} outputs generated")
    return agent_outputs

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    vector_store_info = vector_store.get_collection_info() if vector_store else {}
    return {
        "status": "healthy",
        "llm_provider": "Groq",
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "vector_store": "Simple/In-Memory",
        "vector_store_points": vector_store_info.get('points_count', 0),
        "agents": 6
    }

@app.get("/agents")
async def get_agents():
    """Get list of available agents"""
    return {
        "agents": [
            {"name": "SummaryAgent", "status": "active", "llm": "Groq", "description": "Generates meeting summaries"},
            {"name": "DecisionAgent", "status": "active", "llm": "Groq", "description": "Extracts decisions"},
            {"name": "RiskAgent", "status": "active", "llm": "Groq", "description": "Identifies risks"},
            {"name": "TaskAgent", "status": "active", "llm": "Groq", "description": "Extracts action items"},
            {"name": "WeakSignalAgent", "status": "active", "llm": "Groq", "description": "Detects weak signals"},
            {"name": "KnowledgeGraphAgent", "status": "active", "llm": "Groq", "description": "Builds knowledge graphs"}
        ]
    }

@app.get("/collections")
async def get_collections():
    """Get vector store collections"""
    try:
        if vector_store:
            info = vector_store.get_collection_info()
            return info
        return {"error": "Vector store not initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/vectors")
async def debug_vectors():
    """Debug endpoint to check vector store contents"""
    try:
        if vector_store:
            texts = vector_store.get_all_texts()
            return {
                "total_vectors": len(vector_store.vectors),
                "sample_texts": texts[:5] if texts else []
            }
        return {"error": "Vector store not initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_DEBUG", "True").lower() == "true"
    )