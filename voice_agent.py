import os
import asyncio
from functools import partial
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, RoomInputOptions, Agent
from livekit.agents.llm import function_tool

from livekit.plugins import (
    google,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)

# RAG imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Import our config loader
from config_loader import config

# Load environment variables
load_dotenv(".env")

# ----------------- Global RAG Components -----------------
rag_chain = None
memory = None
vectorstore = None
rag_initialized = asyncio.Event()

async def initialize_rag():
    """Initialize RAG components asynchronously at startup"""
    global rag_chain, memory, vectorstore
    
    print("🔄 Pre-loading RAG system...")
    
    # Get RAG configuration
    rag_config = config.get("agent.rag", {})
    index_path = rag_config.get("vectorstore_path")
    embedding_model_name = rag_config.get("embedding_model")
    retrieval_k = rag_config.get("retrieval_k", 3)
    
    if not index_path:
        raise ValueError("Missing vectorstore_path in configuration")
    
    # Load vectorstore in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    try:
        vectorstore = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                partial(load_vectorstore_sync, index_path, embedding_model_name)
            ),
            timeout=60.0  # 60 second timeout for initial loading
        )
        
        # Build RAG chain
        rag_llm = ChatGoogleGenerativeAI(
            model=rag_config.get("llm_model", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        rag_chain, memory = build_runnable_rag(rag_llm, vectorstore, retrieval_k)
        rag_initialized.set()
        print("✅ RAG system pre-loaded and ready!")
        
    except asyncio.TimeoutError:
        print("❌ RAG initialization timed out. Please check your vectorstore path.")
        rag_initialized.set()  # Still set to avoid blocking forever
    except Exception as e:
        print(f"❌ RAG initialization error: {e}")
        rag_initialized.set()  # Still set to avoid blocking forever

def load_vectorstore_sync(index_path: str, embedding_model_name: str):
    """Synchronous function to load vectorstore (run in thread pool)"""
    print(f"Loading FAISS vector store from: {index_path}")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vs = FAISS.load_local(
        index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded.")
    return vs

def build_runnable_rag(llm, vs, k=3):
    """Build RAG chain with optimized configuration"""
    retriever = vs.as_retriever(search_kwargs={"k": k})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Optimized prompt for faster responses
    prompt_template = """You are the official **RDL Technologies AI Agent**.
Use the following context to answer the user's question. Keep answers concise and focused.

Context: {context}
Chat History: {chat_history}
Question: {question}
Answer:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    rag_chain = (
        RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: memory.chat_memory.messages,
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, memory

@function_tool
async def query_rag_database(question: str) -> str:
    """Query the RDL knowledge base with async optimization and timeout protection"""
    global rag_chain, memory
    
    # Wait for RAG initialization to complete
    if not rag_initialized.is_set():
        print("⏳ RAG system still initializing, please wait...")
        await rag_initialized.wait()
    
    # If RAG failed to initialize, return graceful error
    if rag_chain is None or memory is None:
        return "I apologize, but the knowledge base is currently unavailable. Please try again later."
    
    # Run RAG query in thread pool with strict timeout
    try:
        loop = asyncio.get_event_loop()
        final_answer = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                partial(rag_chain.invoke, question)
            ),
            timeout=15.0  # 15 second timeout for queries (prevents Deepgram disconnects)
        )
        
        memory.save_context({"question": question}, {"answer": final_answer})
        return final_answer
        
    except asyncio.TimeoutError:
        return "I apologize, but the knowledge base query is taking longer than expected. Please try again with a more specific question or ask me to summarize the key points."
    except Exception as e:
        print(f"❌ RAG query error: {e}")
        return "I encountered an error while searching the knowledge base. Please try again with a different question."

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=config.get("agent.instructions"),
            tools=[query_rag_database]
        )

async def entrypoint(ctx: agents.JobContext):
    # Validate required configuration
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

    # Load required API keys
    required_api_keys = ["DEEPGRAM_API_KEY", "GOOGLE_API_KEY", "CARTESIA_API_KEY"]
    for key in required_api_keys:
        if not os.getenv(key):
            print(f"❌ Missing required API key: {key}")
            return

    # Start RAG initialization in background (non-blocking)
    rag_init_task = asyncio.create_task(initialize_rag())

    # Configure components using config
    vad = silero.VAD.load()
    
    # Get model configurations
    stt_config = config.get("agent.models.stt", {})
    tts_config = config.get("agent.models.tts", {})
    llm_config = config.get("agent.models.llm", {})

    # Initialize LLM with region support
    llm = google.LLM(
        model=llm_config.get("model"),
        
    )
    
    session = AgentSession(
        stt=deepgram.STT(
            model=stt_config.get("model"),
            language=stt_config.get("language", "multi")
        ),
        llm=llm,
        tts=cartesia.TTS(
            model=tts_config.get("model"),
            voice=tts_config.get("voice")
        ),
        vad=vad,
    )

    # Configure room input options
    room_input_options = RoomInputOptions()
    if config.get("agent.voice_processing.noise_cancellation", False):
        room_input_options.noise_cancellation = noise_cancellation.BVC()

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=room_input_options,
    )

    print(f"\n{config.get('agent.name', 'Assistant')} is starting up...")
    print("Press [Ctrl+B] to switch to Text mode, then type your questions!\n")

    # Wait for RAG initialization to complete (with timeout)
    try:
        await asyncio.wait_for(rag_init_task, timeout=65.0)
    except asyncio.TimeoutError:
        print("⚠️  RAG initialization taking longer than expected, continuing without it...")

    print("✅ Agent is fully ready!")

    # Initial greeting
    await session.generate_reply(
        instructions="Greet the user and explain that you can answer both general and RDL-specific questions. Mention that the knowledge base is now ready."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))