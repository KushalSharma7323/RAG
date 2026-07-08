import streamlit as st
from dotenv import load_dotenv

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

st.title("📚 PDF RAG Chatbot")
st.caption("Ask questions from your PDF")

# -----------------------------
# Load everything only once
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
        model="mistral-small-2506"
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
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat Box
# -----------------------------
query = st.chat_input("Ask a question...")

if query:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":query
        }
    )

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

    with st.chat_message("assistant"):
        st.markdown(answer)

    with st.expander("Retrieved Context"):

        for i, doc in enumerate(docs, start=1):
            st.markdown(f"### Chunk {i}")
            st.write(doc.page_content)