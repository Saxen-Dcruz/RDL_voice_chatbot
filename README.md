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
LIVEKIT_API_KEY="your_actual_livekit_api_key_here"
LIVEKIT_API_SECRET="your_actual_livekit_secret_here"
GOOGLE_API_KEY="your_actual_google_api_key_here"
DEEPGRAM_API_KEY="your_actual_deepgram_key_here"
CARTESIA_API_KEY="your_actual_cartesia_key_here"
```

# --- 2. KNOWLEDGE BASE SETUP ---
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
## 🎯 How It Works
### Real-time Conversation Flow

1. **Voice Capture**: User speaks into microphone
2. **Speech Recognition**: Deepgram converts speech to text
3. **Intent Analysis**: Google Gemini determines if question is RDL-related
4. **Knowledge Retrieval**: RAG system searches vector database
5. **Context Integration**: Retrieved info + conversation history sent to LLM
6. **Response Generation**: LLM creates natural language answer
7. **Voice Synthesis**: Cartesia converts text response to speech
8. **Audio Output**: User hears the AI's response

### RAG Enhancement
- **Without RAG**: General LLM knowledge only
- **With RAG**: Specific RDL product info, company details, technical specs
