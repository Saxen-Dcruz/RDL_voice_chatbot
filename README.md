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

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root directory with the following variables:

```env
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_secret
GOOGLE_API_KEY=your_google_api_key
DEEPGRAM_API_KEY=your_deepgram_key
CARTESIA_API_KEY=your_cartesia_key
