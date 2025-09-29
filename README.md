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

# 1.1. Create the .env file with placeholder API keys
# NOTE: You MUST replace the 'your_actual_...' placeholders with your real keys.
echo "⚙️ Creating .env file with placeholder keys..."
cat << EOF > .env
LIVEKIT_API_KEY=your_actual_livekit_api_key_here
LIVEKIT_API_SECRET=your_actual_livekit_secret_here
GOOGLE_API_KEY=your_actual_google_api_key_here
DEEPGRAM_API_KEY=your_actual_deepgram_key_here
CARTESIA_API_KEY=your_actual_cartesia_key_here
EOF

# 1.2. Load the environment variables into the current session
echo "✅ Loading environment variables..."
# A simple way to load environment variables from the .env file
export $(grep -v '^#' .env | xargs)

# 1.3. Verify the environment variables (optional, for debugging)
echo "🔍 Verifying GOOGLE_API_KEY loaded: ${GOOGLE_API_KEY:0:4}..." # Shows the first 4 characters

# --- 2. KNOWLEDGE BASE SETUP ---

# 2.1. Placeholder for adding documents. You must manually place files in the folder.
echo "📚 Ensure RDL documents are placed in the 'knowledge_base/' folder."

# 2.2. Process data and build the vector store (FAISS index)
echo "🛠️ Running data ingestion to build the vector database..."
python data_ingestion.py

# --- 3. START APPLICATION ---

# 3.1. Launch the voice agent
echo "🚀 Launching the LiveKit Voice Agent..."
python voice_agent.py console



