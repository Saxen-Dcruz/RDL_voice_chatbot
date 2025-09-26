from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from prompts import RDL_PROMPT
from LLM_setup import load_api_key, initialize_llm

from transformers import pipeline
import asyncio


# ----------------- Exit Intent Detector (Two-Factor) -----------------
exit_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def detect_exit_intent(user_input: str, threshold_low: float = 0.65, threshold_high: float = 0.85):
    normalized = user_input.strip().lower()

    # Immediate exit if keywords used
    if normalized in {"exit", "quit", "bye", "goodbye", "stop", "end"}:
        return "high"

    labels = ["continue_conversation", "end_conversation"]
    result = exit_classifier(user_input, candidate_labels=labels)

    top_label = result["labels"][0]
    top_score = result["scores"][0]

    if top_label == "end_conversation":
        if top_score >= threshold_high:
            return "high"   # Immediate exit
        elif top_score >= threshold_low:
            return "medium" # Ask for confirmation

    return "low"  # Continue


# ----------------- Vector Store Loader -----------------
def load_vectorstore(index_path=r"C:\Users\gerar\Desktop\rdl_data\knowledge_base\faiss_index"):
    print("Loading FAISS vector store............")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = FAISS.load_local(
        index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded.")
    return vectorstore


# ----------------- Build Runnable RAG Chain -----------------
def build_runnable_rag(llm, vectorstore, k=3):
    print("🔧 Building Runnable-based RAG chain...")

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 🧠 Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # 📝 Prompt (with chat history + context)
    prompt = ChatPromptTemplate.from_messages([
        ("system", RDL_PROMPT.template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # 🏗️ Runnable pipeline
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


# ----------------- Chatbot Helper for Streamlit -----------------
def get_chat_response(user_input: str) -> str:
    """
    Function to be used by the Streamlit UI.
    Reuses the same RAG pipeline and memory.
    """
    global rag_chain, memory

    # Initialize chain if not already
    if "rag_chain" not in globals() or "memory" not in globals():
        api_key = load_api_key()
        llm = initialize_llm(api_key)
        vectorstore = load_vectorstore()
        rag_chain, memory = build_runnable_rag(llm, vectorstore, k=4)

    # Check exit intent
    exit_flag = detect_exit_intent(user_input)
    if exit_flag == "high":
        return "Thank you, ending session..."
    elif exit_flag == "medium":
        return "⚠️ Did you mean to end the session? Type 'exit' again to confirm."

    # Normal flow
    final_answer = ""
    for chunk in rag_chain.stream(user_input):  # sync stream for Streamlit
        final_answer += chunk

    # Save context
    memory.save_context({"question": user_input}, {"answer": final_answer})

    return final_answer


# ----------------- Main -----------------
if __name__ == "__main__":
    api_key = load_api_key()
    llm = initialize_llm(api_key)

    vectorstore = load_vectorstore()
    rag_chain, memory = build_runnable_rag(llm, vectorstore, k=4)

    print("\n💬 RDL Assistant ready. Type 'exit' to quit. (Runnable + short-term memory + streaming)")

    async def chat_loop():
        while True:
            query = input("\n Enter your question (or type 'exit' to quit): ").strip()

            # 🔍 Check for exit intent with two-factor logic
            exit_flag = detect_exit_intent(query)

            if exit_flag == "high":
                print("Thank you ,  Ending session...")
                break
            elif exit_flag == "medium":
                confirm = input("⚠️ Did you mean to end the session? Type 'exit' to confirm: ").strip().lower()
                if confirm == "exit":
                    print("✅ Chat session closed. Have a great day!")
                    break
                else:
                    print("👍 Okay, continuing... ask your next question!")
                    continue

            # 🧠 Normal flow
            print("\n🤖 Answer: ", end="", flush=True)

            final_answer = ""
            async for chunk in rag_chain.astream(query):   # ✅ async streaming
                print(chunk, end="", flush=True)
                final_answer += chunk

            print()  # newline
            memory.save_context({"question": query}, {"answer": final_answer})

    asyncio.run(chat_loop())
