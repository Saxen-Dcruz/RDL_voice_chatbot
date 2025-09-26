# voice_agent.py
import os
import asyncio
from functools import partial
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, RoomInputOptions, Agent
from livekit.agents.llm import function_tool
from livekit.plugins import (
    google, cartesia, deepgram, noise_cancellation, silero
)

# Import modular components
from config.config_loader import config
from RAG.rag_chain import RAGManager

# Load environment variables
load_dotenv(".env")

# Global RAG manager (like in your working code)
rag_manager = None

# Define the tool as a global function (like in your working code)
@function_tool
async def query_rag_database(question: str) -> str:
    """Query the RDL knowledge base for information about products, services, or company details.
    
    Args:
        question: The specific question about RDL Technologies to search for
    """
    global rag_manager
    if rag_manager is None:
        return "RAG system is not available yet."
    return await rag_manager.query_rag_database(question)

class Assistant(Agent):
    def __init__(self) -> None:
        instructions = config.get("agent.instructions")
        super().__init__(instructions=instructions, tools=[query_rag_database])

async def entrypoint(ctx: agents.JobContext):
    global rag_manager
    
    # Validate configuration
    try:
        config.validate_required_keys([
            "agent.models.llm.model",
            "agent.models.stt.model", 
            "agent.models.tts.model",
            "agent.rag.vectorstore_path"
        ])
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return

    # Check API keys
    required_api_keys = ["DEEPGRAM_API_KEY", "GOOGLE_API_KEY", "CARTESIA_API_KEY"]
    for key in required_api_keys:
        if not os.getenv(key):
            print(f"❌ Missing required API key: {key}")
            return

    # Initialize RAG manager (global)
    rag_manager = RAGManager()
    rag_init_task = asyncio.create_task(
        rag_manager.initialize_rag(config, os.getenv("GOOGLE_API_KEY"))
    )

    # Configure components
    vad = silero.VAD.load()
    stt_config = config.get("agent.models.stt", {})
    tts_config = config.get("agent.models.tts", {})
    llm_config = config.get("agent.models.llm", {})

    # Create agent session
    session = AgentSession(
        stt=deepgram.STT(
            model=stt_config.get("model"),
            language=stt_config.get("language", "multi")
        ),
        llm=google.LLM(model=llm_config.get("model")),
        tts=cartesia.TTS(
            model=tts_config.get("model"),
            voice=tts_config.get("voice")
        ),
        vad=vad,
    )

    # Configure room options
    room_input_options = RoomInputOptions()
    if config.get("agent.voice_processing.noise_cancellation", False):
        room_input_options.noise_cancellation = noise_cancellation.BVC()

    # Create assistant (no parameters needed now)
    assistant = Assistant()

    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=room_input_options,
    )

    print(f"\n{config.get('agent.name', 'Assistant')} is starting up...")
    print("Press [Ctrl+B] to switch to Text mode\n")

    # Wait for RAG initialization
    try:
        await asyncio.wait_for(rag_init_task, timeout=65.0)
    except asyncio.TimeoutError:
        print("⚠️ RAG initialization taking longer than expected...")

    print("✅ Agent is fully ready!")

    # Initial greeting
    await session.generate_reply(
        instructions="Greet the user and explain you can answer RDL-specific questions."
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))