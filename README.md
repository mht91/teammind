# TeamMind: Multi-Agent Meeting Intelligence System

A sophisticated multi-agent system that analyzes meetings and documents to provide intelligent insights, decisions, risks, and action items.

## Features

- **Multi-modal Input Processing**: Audio, PDF, DOCX, PPTX, Images, Emails, Chats
- **Intelligent Chunking**: Semantic-based document segmentation
- **Hybrid Retrieval**: Combines dense search (embeddings) with sparse search (BM25)
- **Multi-Agent Architecture**:
  - Summary Agent: Generates meeting summaries
  - Decision Agent: Extracts decisions (approved/rejected/postponed)
  - Risk Agent: Identifies potential risks
  - Task Agent: Extracts action items with owners and deadlines
  - Weak Signal Agent: Detects emerging organizational issues
  - Knowledge Graph Agent: Builds relationship graphs
- **Knowledge Graph**: Neo4j for relationship mapping
- **Re-ranking**: BGE Reranker for improved relevance
- **API-first Design**: FastAPI backend with Streamlit/Gradio frontend options

## Tech Stack

| Component | Tool | License |
|-----------|------|---------|
| Speech-to-Text | Whisper | MIT |
| Text Embeddings | BGE/Large | MIT |
| Image Processing | Florence-2 | MIT |
| Vector Database | Qdrant | Apache 2.0 |
| Knowledge Graph | Neo4j | GPLv3 |
| Reranking | BGE Reranker | MIT |
| Orchestration | LangGraph | MIT |
| Backend | FastAPI | MIT |
| Frontend | Streamlit | Apache 2.0 |

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/teammind.git
cd teammind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start services with Docker Compose
docker-compose up -d

# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000