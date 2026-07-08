import streamlit as st
from dotenv import load_dotenv
pip install Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(
    page_title="📚 PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# SIDEBAR (UI Extension)
# -----------------------------
with st.sidebar:
    st.title("⚙️ RAG Settings")
    st.markdown("---")
    st.markdown("**Active Document:**")
    st.code("keip108.pdf")
    st.markdown("**Vector Store:**")
    st.code("Chroma DB (MMR)")
    st.markdown("---")
    st.info("💡 Pro-Tip: The model is configured to reply in simple Hinglish with mnemonics!")
    
    # Quick clear button for convenience
    if st.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# MAIN APP HEADER
# -----------------------------
st.title("📚 PDF RAG Chatbot")
st.caption("Ask questions from your PDF")
st.markdown("---")

# Layout Split: Chat window and context viewing panel
tab_chat, tab_source = st.tabs(["💬 Chat Dashboard", "🔍 Reference Context"])

# -----------------------------
# Load everything only once (Original Function)
# -----------------------------
@st.cache_resource
def initialize_rag():

    loader = PyPDFLoader("RAG/document_loaders/keip108.pdf")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = loader.load()
    docs = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings()

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506",
        max_new_tokens = 100
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant.
Use easy professional language.
Prefer Hinglish wherever possible.
Summarize in bullet points.
Also give mnemonics to remember the topic.
"""
            ),

            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    return retriever, prompt, llm


retriever, prompt, llm = initialize_rag()

# -----------------------------
# Chat History (Placed into Chat Tab)
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# -----------------------------
# Chat Box (Original Logic)
# -----------------------------
query = st.chat_input("Ask a question...")

if query:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":query
        }
    )

    with tab_chat:
        with st.chat_message("user"):
            st.markdown(query)

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke(
        {
            "context":context,
            "question":query
        }
    )

    with st.spinner("Thinking..."):
        response = llm.invoke(final_prompt)

    answer = response.content

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )

    with tab_chat:
        with st.chat_message("assistant"):
            st.markdown(answer)

    # -----------------------------
    # Context Display (Placed into Source Tab)
    # -----------------------------
    with tab_source:
        st.write("### Chunks retrieved for your last question:")
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"#### Chunk {i}")
            st.info(doc.page_content)
            
    # Soft refresh to keep the Tab layout clean and synced
    st.rerun()

with tab_source:
    if not query and not st.session_state.messages:
        st.write("No queries processed yet. Ask a question to inspect document references here.")