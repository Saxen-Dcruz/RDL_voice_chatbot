RDL Chatbot - Voice AI Assistant
A sophisticated voice-enabled AI chatbot for RDL Technologies that combines LiveKit's real-time communication with RAG (Retrieval Augmented Generation) for intelligent, context-aware responses.

🚀 Quick Start
Prerequisites
Python 3.8+

API keys for:

LiveKit

Google Gemini

Deepgram

Cartesia

RDL knowledge base documents

Installation
Clone and setup environment:

bash
git clone <repository>
cd RDL_CHATBOT
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
Configure environment variables:
Create .env file:

env
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_secret
GOOGLE_API_KEY=your_google_api_key
DEEPGRAM_API_KEY=your_deepgram_key
CARTESIA_API_KEY=your_cartesia_key
Prepare knowledge base:

Place RDL documents in knowledge_base/ folder

Run data ingestion:

bash
python data_ingestion.py
Start the voice agent:

bash
python voice_agent.py
📁 File Structure & Components
text
RDL_CHATBOT/
├── agents/
│   ├── assistant.py          # Main agent class with RAG tool integration
│   └── __init__.py
├── config/
│   ├── config_loader.py      # Configuration management
│   ├── config.yaml          # Main configuration file
│   └── __init__.py
├── knowledge_base/
│   └── faiss_index/         # Vector database (auto-generated)
├── RAG/
│   ├── rag_chain.py         # RAG system implementation
│   └── __init__.py
├── data_ingestion.py        # Knowledge base processing
├── voice_agent.py          # Main application entry point
├── prompts.py              # LLM prompt templates
└── requirements.txt        # Dependencies
🔧 Detailed Component Analysis
1. Configuration System (config/)
config_loader.py

Purpose: Centralized configuration management with environment variable resolution

Key Functions:

_load_config(): Loads YAML configuration

_resolve_env_vars(): Replaces ${VARIABLE} with actual environment values

get(): Dot-notation access to nested config values

validate_required_keys(): Ensures essential configuration is present

config.yaml

LiveKit connection settings

AI model configurations (Gemini, Deepgram, Cartesia)

RAG parameters (vector store path, embedding model)

Voice processing settings

2. RAG System (RAG/rag_chain.py)
RAGManager Class - Core intelligent retrieval system:

Initialization Flow:

python
async def initialize_rag()
├── Load vector store asynchronously
├── Initialize embedding model (BAAI/bge-m3)
├── Setup conversation memory
└── Build LangChain RAG pipeline
Query Processing:

python
async def query_rag_database(question)
├── Wait for RAG initialization
├── Retrieve relevant documents using FAISS
├── Generate context-aware response using Gemini
├── Update conversation memory
└── Return formatted answer
Key Features:

Asynchronous loading for faster startup

Conversation memory for contextual responses

Timeout handling for reliability

Optimized document retrieval with configurable k-value

3. AI Agent (agents/assistant.py)
Assistant Class - Extends LiveKit Agent:

python
class Assistant(Agent):
    def __init__(self, rag_manager, config)
    └── @llm.tool() query_rag_database(question)
        └── Calls rag_manager.query_rag_database()
Tool Integration: Decorates RAG query function as an LLM tool, allowing the agent to automatically use knowledge base when RDL-related questions are detected.

4. Data Processing (data_ingestion.py)
Knowledge Base Pipeline:

python
main()
├── load_documents_from_directory()
├── split_document_by_products()  # Product-level chunking
├── get_embedding_model()        # HuggingFace embeddings
└── FAISS.from_documents()       # Vector database creation
Chunking Strategy: Uses product boundaries (###**Product Name**###) to create semantically meaningful document chunks.

5. Voice Agent (voice_agent.py)
Main Application Entry Point:

Startup Sequence:

python
async def entrypoint(ctx)
├── Validate configuration and API keys
├── Initialize RAG manager (background task)
├── Setup voice components (STT, TTS, VAD)
├── Create agent session with LiveKit room
├── Send initial greeting
└── Wait for RAG initialization completion
Voice Components:

STT (Speech-to-Text): Deepgram for accurate transcription

TTS (Text-to-Speech): Cartesia for natural voice output

VAD (Voice Activity Detection): Silero for detecting speech

Noise Cancellation: BVC for audio quality

🔄 System Flowchart





















🎯 How It Works
Real-time Conversation Flow:
Voice Capture: User speaks through microphone

Speech Recognition: Deepgram converts speech to text

Intent Analysis: Google Gemini determines if question is RDL-related

Knowledge Retrieval: For RDL questions, RAG system searches vector database

Context Integration: Retrieved information + conversation history sent to LLM

Response Generation: LLM creates natural language answer

Voice Synthesis: Cartesia converts text response to speech

Audio Output: User hears the AI's response

RAG Enhancement:
Without RAG: General LLM knowledge only

With RAG: Specific RDL product information, company details, technical specifications

⚡ Performance Optimizations
Asynchronous Initialization: RAG loads in background while agent starts

Connection Pooling: Reusable API connections

Memory Management: Conversation buffer with configurable limits

Timeout Handling: Graceful degradation under load

Error Recovery: Automatic retries for transient failures

🔧 Customization
Adding New Knowledge:
Add documents to knowledge_base/ folder

Run python data_ingestion.py

Restart voice agent

Modifying Behavior:
Edit config.yaml for model settings

Update prompts.py for response formatting

Modify agents/assistant.py for tool behavior

This system provides a robust, scalable voice AI assistant specifically tuned for RDL Technologies with intelligent knowledge retrieval capabilities.
