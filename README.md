# RDL Chatbot - Voice AI Assistant

A sophisticated voice-enabled AI chatbot for **RDL Technologies** that combines **LiveKit's real-time communication** with **RAG (Retrieval Augmented Generation)** for intelligent, context-aware responses.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API keys for:
  - LiveKit
  - Google Gemini
  - Deepgram
  - Cartesia
- RDL knowledge base documents

### Installation
Clone and setup environment:

```bash
git clone https://github.com/Saxen-Dcruz/RDL_voice_chatbot.git
cd RDL_CHATBOT
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt 
```
#!/bin/bash

# --- 1. ENVIRONMENT SETUP ---

# 1. Create the .env file with placeholder API keys
# NOTE: You MUST replace the 'your_actual_...' placeholders with your real keys.
echo "⚙️ Creating .env file with placeholder keys..."
cat << EOF > .env
LIVEKIT_API_KEY=your_actual_livekit_api_key_here
LIVEKIT_API_SECRET=your_actual_livekit_secret_here
GOOGLE_API_KEY=your_actual_google_api_key_here
DEEPGRAM_API_KEY=your_actual_deepgram_key_here
CARTESIA_API_KEY=your_actual_cartesia_key_here
EOF



# --- 2. KNOWLEDGE BASE SETUP ---

# 2.1. Placeholder for adding documents. You must manually place files in the folder.
e"📚 Ensure RDL documents are placed in the 'knowledge_base/' folder."

# 2.2. Process data and build the vector store (FAISS index)
echo "🛠️ Running data ingestion to build the vector database..."
python data_ingestion.py

# --- 3. START APPLICATION ---

# 3.1. Launch the voice agent
echo "🚀 Launching the LiveKit Voice Agent..."
python voice_agent.py console


graph TD
    A[User Voice Input] --> B[LiveKit Room]
    B --> C[Deepgram STT]
    C --> D[Text Transcription]
    D --> E[Google LLM Analysis]
    
    E --> F{RDL Question?}
    F -->|Yes| G[query_rag_database Tool]
    F -->|No| H[Direct LLM Response]
    
    G --> I[RAG Manager]
    I --> J[FAISS Vector Search]
    J --> K[Context Retrieval]
    K --> L[Gemini Context+Question]
    L --> M[Knowledge Base Answer]
    
    M --> E
    H --> N[Cartesia TTS]
    M --> N
    N --> O[Voice Response]
    O --> P[User Hearing]
    
    Q[Knowledge Base] --> R[Data Ingestion]
    R --> S[FAISS Index]
    S --> J


