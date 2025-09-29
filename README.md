# RDL Chatbot - Voice AI Assistant

A sophisticated voice-enabled AI chatbot for **RDL Technologies** that combines **LiveKit's real-time communication** with **RAG (Retrieval Augmented Generation)** for intelligent, context-aware responses.

---

### Prerequisites
- Python 3.9+
- API keys for:
  - LiveKit
  - Google Gemini
  - Deepgram
  - Cartesia
- RDL knowledge base documents

# Installation
## Clone and setup environment:
```bash
git clone https://github.com/Saxen-Dcruz/RDL_voice_chatbot.git
cd RDL_CHATBOT
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt 
```
# --- 1. ENVIRONMENT SETUP ---
## 1. Create the .env file with placeholder API keys
#### NOTE: You MUST replace the 'your_actual_...' placeholders with your real keys.
```env
LIVEKIT_API_KEY=your_actual_livekit_api_key_here
LIVEKIT_API_SECRET=your_actual_livekit_secret_here
GOOGLE_API_KEY=your_actual_google_api_key_here
DEEPGRAM_API_KEY=your_actual_deepgram_key_here
CARTESIA_API_KEY=your_actual_cartesia_key_here
```

# --- 2. KNOWLEDGE BASE SETUP ---
## Placeholder for adding documents. You must manually place files in the folder.
### Ensure RDL documents are placed in the 'knowledge_base/' folder.

## Process data and build the vector store (FAISS index)
   Run data ingestion to build the FAISS vector database...
   ```bash
   python data_ingestion.py
```
# --- 3. START APPLICATION ---
## Launch the voice agent
``` bash
python voice_agent.py console
```


