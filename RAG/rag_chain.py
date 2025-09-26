import asyncio
from functools import partial
from typing import Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGManager:
    def __init__(self):
        self.rag_chain: Optional[RunnableParallel] = None
        self.memory: Optional[ConversationBufferMemory] = None
        self.vectorstore: Optional[FAISS] = None
        self.rag_initialized = asyncio.Event()
        self._initialization_lock = asyncio.Lock()

    async def initialize_rag(self, config, google_api_key) -> None:
        """Initialize RAG components asynchronously at startup with optimizations"""
        async with self._initialization_lock:
            if self.rag_initialized.is_set():
                return
                
            print("🔄 Pre-loading RAG system...")

            rag_config = config.get("agent.rag", {})
            index_path = rag_config.get("vectorstore_path")
            embedding_model_name = rag_config.get("embedding_model", "BAAI/bge-m3")
            retrieval_k = rag_config.get("retrieval_k", 3)

            if not index_path:
                raise ValueError("Missing vectorstore_path in configuration")

            try:
                # Load vectorstore with timeout
                self.vectorstore = await asyncio.wait_for(
                    self._load_vectorstore_async(index_path, embedding_model_name),
                    timeout=60.0
                )

                # Initialize LLM and chain
                rag_llm = ChatGoogleGenerativeAI(
                    model=rag_config.get("llm_model"),
                    google_api_key=google_api_key,
                    temperature=0.1,
                    max_retries=2
                )
                
                self.rag_chain, self.memory = await asyncio.get_event_loop().run_in_executor(
                    None, self._build_runnable_rag, rag_llm, self.vectorstore, retrieval_k
                )
                
                self.rag_initialized.set()
                print("✅ RAG system pre-loaded and ready!")

            except asyncio.TimeoutError:
                print("❌ RAG initialization timed out.")
                self.rag_initialized.set()
            except Exception as e:
                print(f"❌ RAG initialization error: {e}")
                self.rag_initialized.set()

    async def _load_vectorstore_async(self, index_path: str, embedding_model_name: str) -> FAISS:
        """Asynchronously load vectorstore with better error handling"""
        print(f"📚 Loading FAISS vector store from: {index_path}")
        
        # CORRECTED: Use partial to pass model_name as keyword argument
        embedding_model = await asyncio.get_event_loop().run_in_executor(
            None,
            partial(HuggingFaceEmbeddings, model_name=embedding_model_name)
        )
        
        # Load vectorstore
        vectorstore = await asyncio.get_event_loop().run_in_executor(
            None,
            partial(
                FAISS.load_local,
                index_path,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True
            )
        )
        
        print("✅ Vector store loaded.")
        return vectorstore

    def _build_runnable_rag(self, llm: ChatGoogleGenerativeAI, vs: FAISS, k: int = 3) -> Tuple[RunnableParallel, ConversationBufferMemory]:
        """Build optimized RAG chain"""
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "fetch_k": min(20, k * 3)
            }
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Make sure RDL_PROMPT is properly defined
        from prompts import RDL_PROMPT
        
        rag_chain = (
            RunnableParallel({
                "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
                "question": RunnablePassthrough(),
            })
            | RDL_PROMPT
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, memory

    async def query_rag_database(self, question: str) -> str:
        """Optimized RAG query with better error handling and caching"""
        try:
            await asyncio.wait_for(self.rag_initialized.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            return "RAG system is still initializing. Please try again in a moment."

        if self.rag_chain is None or self.memory is None:
            return "I apologize, but the knowledge base is currently unavailable."

        if not question or len(question.strip()) < 2:
            return "Please provide a more specific question."

        try:
            loop = asyncio.get_event_loop()
            final_answer = await asyncio.wait_for(
                loop.run_in_executor(None, partial(self._safe_chain_invoke, question)),
                timeout=10.0
            )
            
            if final_answer and len(final_answer.strip()) > 10:
                await loop.run_in_executor(
                    None, 
                    partial(self.memory.save_context, {"question": question}, {"answer": final_answer})
                )
            
            return final_answer
            
        except asyncio.TimeoutError:
            return "The query is taking longer than expected. Please try a more specific question."
        except Exception as e:
            print(f"❌ RAG query error: {e}")
            return "I encountered an error while searching the knowledge base. Please try again."

    def _safe_chain_invoke(self, question: str) -> str:
        """Safely invoke the chain with error handling"""
        try:
            return self.rag_chain.invoke(question)
        except Exception as e:
            print(f"❌ Chain invocation error: {e}")
            return f"I encountered an error: {str(e)}"

    async def get_conversation_history(self) -> list:
        """Get current conversation history"""
        if self.memory:
            return self.memory.chat_memory.messages
        return []

    async def clear_memory(self) -> None:
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()

    def is_ready(self) -> bool:
        """Check if RAG system is ready"""
        return self.rag_initialized.is_set() and self.rag_chain is not None